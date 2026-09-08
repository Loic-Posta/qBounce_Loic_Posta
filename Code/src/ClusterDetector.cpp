#include "ClusterDetector.h"
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cmath>

// ─── Cluster helpers ─────────────────────────────────────────────────────────

std::vector<float> Cluster::toFeatureVector() const
{
    return {
        static_cast<float>(size_px),
        aspect_ratio,
        elongation,
        compactness,
        total_charge,
        peak_value,
        peak_snr,
        sigma_x,
        sigma_y,
        sigma_x / (sigma_y + 1e-6f)   // axis ratio
    };
}

std::string Cluster::csvHeader()
{
    // 20 columns — must stay in sync with toCsvLine() below.
    return "image_id,cluster_id,"
           "center_x,center_y,"
           "bbox_x,bbox_y,bbox_w,bbox_h,"
           "size_px,aspect_ratio,elongation,compactness,"
           "total_charge,peak_value,peak_snr,"
           "sigma_x,sigma_y,"
           "label,confidence,"
           "filter_status";
}

std::string Cluster::toCsvLine() const
{
    auto labelStr = [](Label l) -> std::string {
        switch(l) {
            case Label::PARTICLE:   return "particle";
            case Label::ARTIFACT:   return "artifact";
            case Label::COSMIC_RAY: return "cosmic_ray";
            case Label::HOT_PIXEL:  return "hot_pixel";
            default:                return "unknown";
        }
    };

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3)
       << image_id              << ","   // col  1
       << cluster_id            << ","   // col  2
       << center_x              << ","   // col  3
       << center_y              << ","   // col  4
       << bounding_box.x        << ","   // col  5
       << bounding_box.y        << ","   // col  6
       << bounding_box.width    << ","   // col  7
       << bounding_box.height   << ","   // col  8
       << size_px               << ","   // col  9
       << aspect_ratio          << ","   // col 10
       << elongation            << ","   // col 11
       << compactness           << ","   // col 12
       << total_charge          << ","   // col 13
       << peak_value            << ","   // col 14
       << peak_snr              << ","   // col 15
       << sigma_x               << ","   // col 16
       << sigma_y               << ","   // col 17
       << labelStr(label)       << ","   // col 18
       << confidence            << ","   // col 19
       << (is_valid ? "valid" : "rejected");  // col 20
    return ss.str();
}

// ─── ClusterDetector ─────────────────────────────────────────────────────────

std::vector<Cluster> ClusterDetector::detect(const cv::Mat& event_mask,
                                              const cv::Mat& raw,
                                              const cv::Mat& mean,
                                              const cv::Mat& sigma,
                                              int            image_id)
{
    // ── Morphological closing: bridge small gaps in split tracks ──────────
    // A minimum-ionising track (e.g. a muon crossing at a shallow angle)
    // can dip below n_sigma for a pixel or two — read-out noise, a single
    // under-threshold pixel, a dead pixel excluded from the event mask —
    // and connectedComponentsWithStats() then splits what is physically
    // one track into two-or-more small clusters, each with its own
    // (misleadingly low) size_px / total_charge / elongation.
    //
    // Closing (dilate, then erode) with a small structuring element bridges
    // gaps up to ~cluster_gap_px_ pixels wide: dilate() grows every
    // foreground region enough to jump the gap and merge, then erode()
    // shrinks the result back down by the same amount, restoring the
    // original outer footprint everywhere except inside the newly-bridged
    // gaps. cluster_gap_px_ == 0 (the default) is a strict no-op — the
    // mask that reaches connectedComponentsWithStats() is bit-identical to
    // event_mask, so existing behaviour is unchanged unless this is
    // explicitly enabled.
    cv::Mat closed_mask;
    if (cluster_gap_px_ > 0) {
        const int k = 2 * cluster_gap_px_ + 1;   // odd kernel size spanning the gap
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        cv::morphologyEx(event_mask, closed_mask, cv::MORPH_CLOSE, kernel);
    } else {
        closed_mask = event_mask;
    }

    // Connected-components labeling (8-connectivity)
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(closed_mask, labels, stats, centroids,
                                             8, CV_32S);

    std::vector<Cluster> clusters;
    clusters.reserve(n - 1);   // label 0 = background

    // Runtime floor takes precedence over the compile-time one when higher —
    // filtering on the cheap connected-components area BEFORE the expensive
    // characterise() call is what makes this a throughput lever, not just a
    // CSV-size lever.
    const int min_size = std::max(MIN_CLUSTER_SIZE, min_size_px_);

    for (int i = 1; i < n; ++i) {
        int sz = stats.at<int>(i, cv::CC_STAT_AREA);

        // Quick size pre-filter before full characterisation
        if (sz < min_size || sz > MAX_CLUSTER_SIZE) continue;

        // Collect pixel coordinates for this label
        cv::Rect bbox(stats.at<int>(i, cv::CC_STAT_LEFT),
                      stats.at<int>(i, cv::CC_STAT_TOP),
                      stats.at<int>(i, cv::CC_STAT_WIDTH),
                      stats.at<int>(i, cv::CC_STAT_HEIGHT));

        std::vector<cv::Point> coords;
        coords.reserve(sz);
        // Only iterate within the bounding box for speed
        cv::Mat roi_labels = labels(bbox);
        for (int r = 0; r < roi_labels.rows; ++r) {
            const int* row = roi_labels.ptr<int>(r);
            for (int c = 0; c < roi_labels.cols; ++c) {
                if (row[c] == i)
                    coords.emplace_back(bbox.x + c, bbox.y + r);
            }
        }

        auto footprint = growFootprint(i, coords, labels, raw, mean, sigma);

        Cluster cl = characterise(i, image_id, footprint, raw, mean, sigma);
        if (!footprint.empty()) {
            cl.bounding_box = cv::boundingRect(footprint);
        } else {
            cl.bounding_box = bbox;
        }
        cl.pixel_coords = std::move(footprint);   // retain exact footprint (post characterise(), safe to move)

        // Charge floor: needs total_charge, so it runs after characterise().
        // Below-floor clusters are dropped, not just flagged — the point is
        // to keep them out of the CSV/pipe/classifier entirely.
        if (min_charge_adu_ > 0.f && cl.total_charge < min_charge_adu_) continue;

        if (cl.peak_snr < MIN_PEAK_SNR) {
            cl.is_valid = false;
        }

        clusters.push_back(std::move(cl));
    }

    return clusters;
}

// ─── private ─────────────────────────────────────────────────────────────────

std::vector<cv::Point> ClusterDetector::growFootprint(
    int cluster_label,
    const std::vector<cv::Point>& coords,
    const cv::Mat& labels,
    const cv::Mat& raw,
    const cv::Mat& mean,
    const cv::Mat& sigma) const
{
    if (cluster_grow_px_ <= 0 || coords.empty()) {
        return coords;
    }

    cv::Mat included = cv::Mat::zeros(raw.size(), CV_8UC1);
    std::vector<cv::Point> grown = coords;
    grown.reserve(coords.size() * (2 * cluster_grow_px_ + 1));

    for (const auto& p : coords) {
        included.at<uint8_t>(p) = 255;
    }

    for (const auto& p : coords) {
        for (int dy = -cluster_grow_px_; dy <= cluster_grow_px_; ++dy) {
            for (int dx = -cluster_grow_px_; dx <= cluster_grow_px_; ++dx) {
                if (dx == 0 && dy == 0) continue;

                cv::Point q(p.x + dx, p.y + dy);
                if (q.x < 0 || q.x >= raw.cols || q.y < 0 || q.y >= raw.rows) {
                    continue;
                }
                if (included.at<uint8_t>(q)) {
                    continue;
                }

                int neighbour_label = labels.at<int>(q);
                if (neighbour_label != 0 && neighbour_label != cluster_label) {
                    continue;  // never steal pixels from another detected component
                }

                float raw_val = static_cast<float>(raw.at<uint8_t>(q));
                float bg_mean = mean.at<float>(q);
                float bg_sigma = std::max(sigma.at<float>(q), 1.0f);
                float snr = (raw_val - bg_mean) / bg_sigma;

                if (snr >= cluster_grow_min_snr_) {
                    included.at<uint8_t>(q) = 255;
                    grown.push_back(q);
                }
            }
        }
    }

    return grown;
}

Cluster ClusterDetector::characterise(int                           cluster_id,
                                       int                           image_id,
                                       const std::vector<cv::Point>& coords,
                                       const cv::Mat&                raw,
                                       const cv::Mat&                mean,
                                       const cv::Mat&                sigma)
{
    Cluster cl;
    cl.image_id   = image_id;
    cl.cluster_id = cluster_id;
    cl.size_px    = static_cast<int>(coords.size());

    // ── Photometry & charge-weighted centroid ─────────────────────────────
    //
    // peak_val and peak_snr are tracked INDEPENDENTLY.
    //
    // Rationale: when multiple pixels are saturated at 255 AND the background
    // model has drifted to ~255 (e.g. a persistently bright region), the SNR
    // computed at the brightest pixel is (255-255)/sigma ≈ 0, which is wrong.
    // max_snr scans every pixel and keeps the highest (raw-bg)/sigma found,
    // which correctly identifies the best-measured pixel in the cluster even
    // if it is not the one with the highest raw ADU value.
    double sum_charge = 0.0;
    double sum_cx     = 0.0;
    double sum_cy     = 0.0;
    float  peak_val   = 0.f;   // highest raw ADU in cluster
    float  max_snr    = 0.f;   // highest per-pixel SNR in cluster (independent)

    for (const auto& p : coords) {
        float raw_val  = static_cast<float>(raw.at<uint8_t>(p));
        float bg_mean  = mean.at<float>(p);
        float bg_sigma = sigma.at<float>(p);
        float charge   = raw_val - bg_mean;

        // Per-pixel SNR: excess above background in units of local noise.
        // Use a minimum sigma floor of 1.0 to avoid division-by-zero when the
        // background model has not yet converged on a pixel.
        float effective_sigma = std::max(bg_sigma, 1.0f);
        float cur_snr         = charge / effective_sigma;

        if (charge < 0.f) charge = 0.f;   // floor charge at zero

        sum_charge += charge;
        sum_cx     += charge * p.x;
        sum_cy     += charge * p.y;

        // Track peak raw value (for display / photometry)
        if (raw_val > peak_val)
            peak_val = raw_val;

        // Track peak SNR independently — this is the reported cluster SNR
        if (cur_snr > max_snr)
            max_snr = cur_snr;
    }

    cl.total_charge = static_cast<float>(sum_charge);
    cl.peak_value   = peak_val;
    cl.peak_snr     = max_snr;   // maximum per-pixel SNR across the cluster

    if (sum_charge > 0.0) {
        cl.center_x = static_cast<float>(sum_cx / sum_charge);
        cl.center_y = static_cast<float>(sum_cy / sum_charge);
    } else {
        // Fallback: geometric centroid
        double gx = 0, gy = 0;
        for (const auto& p : coords) { gx += p.x; gy += p.y; }
        cl.center_x = static_cast<float>(gx / coords.size());
        cl.center_y = static_cast<float>(gy / coords.size());
    }

    // ── Shape: bounding-box aspect ratio & compactness ────────────────────
    int xmin = coords[0].x, xmax = coords[0].x;
    int ymin = coords[0].y, ymax = coords[0].y;
    for (const auto& p : coords) {
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
    }
    int bbox_w = xmax - xmin + 1;
    int bbox_h = ymax - ymin + 1;
    cl.aspect_ratio = (bbox_h > 0) ? static_cast<float>(bbox_w) / bbox_h : 1.f;
    cl.compactness  = static_cast<float>(cl.size_px) / static_cast<float>(bbox_w * bbox_h);

    // ── Shape: charge-weighted RMS spread (σx, σy) ────────────────────────
    double sum_dx2 = 0.0, sum_dy2 = 0.0;
    for (const auto& p : coords) {
        float raw_val = static_cast<float>(raw.at<uint8_t>(p));
        float bg_mean = mean.at<float>(p);
        float charge  = std::max(0.f, raw_val - bg_mean);
        double dx = p.x - cl.center_x;
        double dy = p.y - cl.center_y;
        sum_dx2 += charge * dx * dx;
        sum_dy2 += charge * dy * dy;
    }
    if (sum_charge > 0.0) {
        cl.sigma_x = static_cast<float>(std::sqrt(sum_dx2 / sum_charge));
        cl.sigma_y = static_cast<float>(std::sqrt(sum_dy2 / sum_charge));
    }

    // ── Shape: elongation via inertia tensor eigenvalue ratio ─────────────
    // Ixx, Iyy, Ixy (charge-weighted second moments)
    double Ixx = 0, Iyy = 0, Ixy = 0;
    for (const auto& p : coords) {
        float raw_val = static_cast<float>(raw.at<uint8_t>(p));
        float bg_mean = mean.at<float>(p);
        float charge  = std::max(0.f, raw_val - bg_mean);
        double dx = p.x - cl.center_x;
        double dy = p.y - cl.center_y;
        Ixx += charge * dy * dy;
        Iyy += charge * dx * dx;
        Ixy += charge * dx * dy;
    }
    // Eigenvalues of 2×2 symmetric matrix
    double trace  = Ixx + Iyy;
    double det    = Ixx * Iyy - Ixy * Ixy;
    double disc   = std::sqrt(std::max(0.0, trace * trace / 4.0 - det));
    double lam1   = trace / 2.0 + disc;
    double lam2   = trace / 2.0 - disc;
    cl.elongation = (lam2 > 1e-6) ? static_cast<float>(lam1 / lam2) : 1.f;

    return cl;
}
