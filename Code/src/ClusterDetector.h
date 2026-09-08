#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @brief All features extracted from a single detected cluster.
 *
 * These features are sufficient for:
 *   - basic counting (just use `is_valid`)
 *   - manual quality cuts (size, charge, aspect ratio)
 *   - ML classification input (all float fields → feature vector)
 */
struct Cluster {
    // ── Identity ───────────────────────────────────────────────────────────
    int   image_id   = 0;
    int   cluster_id = 0;

    // ── Spatial ────────────────────────────────────────────────────────────
    float center_x   = 0.f;   // subpixel centroid (charge-weighted)
    float center_y   = 0.f;
    cv::Rect bounding_box;     // tight bounding box in full image coords

    // ── Size & Shape ───────────────────────────────────────────────────────
    int   size_px         = 0;   // number of pixels in cluster
    float aspect_ratio    = 0.f; // width / height of bounding box
    float elongation      = 0.f; // eigenvalue ratio of inertia tensor (1=round, >1=elongated)
    float compactness     = 0.f; // size_px / (bbox_w * bbox_h)  [0..1]

    // ── Photometry ─────────────────────────────────────────────────────────
    float total_charge    = 0.f; // sum of (pixel - background) over cluster
    float peak_value      = 0.f; // max raw pixel value in cluster
    float peak_snr        = 0.f; // (peak_value - mean_bg) / sigma_bg at peak pixel

    // ── Radial Profile ─────────────────────────────────────────────────────
    float sigma_x         = 0.f; // RMS spread in x (charge-weighted)
    float sigma_y         = 0.f; // RMS spread in y (charge-weighted)

    // ── Classification result (set by ML classifier if used) ───────────────
    enum class Label { UNKNOWN, PARTICLE, ARTIFACT, COSMIC_RAY, HOT_PIXEL };
    Label label           = Label::UNKNOWN;
    float confidence      = 0.f;   // classifier confidence [0..1]

    // ── Validity flag ──────────────────────────────────────────────────────
    bool  is_valid        = true;  // false → rejected by quality cuts

    // ── Pixel footprint ────────────────────────────────────────────────────
    // Exact member-pixel coordinates (full-image coords), as found by the
    // connected-components pass in ClusterDetector::detect(). This is the
    // one field NOT part of the CSV output (csvHeader()/toCsvLine() are
    // unchanged) — it exists purely so downstream consumers (e.g. a data
    // exporter) can build a pixel-exact mask instead of falling back to the
    // coarser bounding_box, which generally includes background pixels the
    // cluster doesn't actually cover.
    std::vector<cv::Point> pixel_coords;

    /** Return features as a flat float vector for ML input */
    std::vector<float> toFeatureVector() const;

    /** CSV header matching toFeatureVector() */
    static std::string csvHeader();

    /** Serialise to one CSV line */
    std::string toCsvLine() const;
};

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Extracts and characterises clusters from an event mask.
 */
class ClusterDetector {
public:
    // ── Quality cuts ───────────────────────────────────────────────────────
    static constexpr int   MIN_CLUSTER_SIZE  = 1;    // px
    static constexpr int   MAX_CLUSTER_SIZE  = 500;  // px  (larger → likely artefact)
    static constexpr float MIN_PEAK_SNR      = 3.0f;

    /**
     * @param cluster_gap_px  morphological-closing radius (px) applied to the
     *                        event mask before connected-components labeling,
     *                        to bridge small gaps in split tracks (e.g. a
     *                        muon track broken by a couple of under-threshold
     *                        pixels). 0 (default) disables closing entirely —
     *                        the mask passed to connectedComponentsWithStats()
     *                        is then bit-identical to the input event_mask.
     * @param cluster_grow_px extra radius (px) used after connected-components
     *                        labeling to include weaker neighbouring pixels in
     *                        the same cluster footprint. This does not merge
     *                        far-away clusters; it only expands each already
     *                        detected component locally.
     * @param cluster_grow_min_snr minimum per-pixel SNR required for a grown
     *                             neighbour pixel to be added.
     * @param min_size_px    clusters smaller than this are DROPPED entirely
     *                       (not even written to the CSV). 0 = keep the old
     *                       behaviour. Purpose: suppress the flood of 1-3 px
     *                       noise/gamma/beta clusters at the source when only
     *                       heavy ion tracks (alpha / Li) are of interest —
     *                       they cost clustering + CSV + classification time
     *                       for nothing.
     * @param min_charge_adu clusters whose total_charge is below this are
     *                       dropped after characterisation. 0 = keep all.
     */
    explicit ClusterDetector(int cluster_gap_px = 0,
                             int cluster_grow_px = 0,
                             float cluster_grow_min_snr = 1.0f,
                             int min_size_px = 0,
                             float min_charge_adu = 0.f)
        : cluster_gap_px_(cluster_gap_px),
          cluster_grow_px_(cluster_grow_px),
          cluster_grow_min_snr_(cluster_grow_min_snr),
          min_size_px_(min_size_px),
          min_charge_adu_(min_charge_adu) {}

    /**
     * @param event_mask  CV_8UC1, 255 = event pixel
     * @param raw         original 8-bit image (for photometry)
     * @param mean        background mean image (CV_32FC1)
     * @param sigma       background sigma image (CV_32FC1)
     * @param image_id    frame counter (for logging)
     */
    std::vector<Cluster> detect(const cv::Mat& event_mask,
                                const cv::Mat& raw,
                                const cv::Mat& mean,
                                const cv::Mat& sigma,
                                int            image_id);

private:
    Cluster characterise(int                    cluster_id,
                         int                    image_id,
                         const std::vector<cv::Point>& coords,
                         const cv::Mat&         raw,
                         const cv::Mat&         mean,
                         const cv::Mat&         sigma);

    std::vector<cv::Point> growFootprint(int cluster_label,
                                         const std::vector<cv::Point>& coords,
                                         const cv::Mat& labels,
                                         const cv::Mat& raw,
                                         const cv::Mat& mean,
                                         const cv::Mat& sigma) const;

    // ── runtime parameter (set once in constructor) ─────────────────────────
    int cluster_gap_px_ = 0;
    int cluster_grow_px_ = 0;
    float cluster_grow_min_snr_ = 1.0f;
    int min_size_px_ = 0;
    float min_charge_adu_ = 0.f;
};
