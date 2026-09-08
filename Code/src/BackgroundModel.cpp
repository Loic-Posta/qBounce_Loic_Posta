#include "BackgroundModel.h"
#include <stdexcept>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <filesystem>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace {
// On-disk model format (binary, little-endian):
//   "BGMD" | int32 version | int32 image_count | float n_sigma
//   | int32 dead_pixel_warmup | int32 rows | int32 cols
//   | mean(rows*cols f32) | M2(f32) | count(f32) | dead_mask(rows*cols u8)
//
// Rationale: at 20 MP a YAML (text) dump of three float matrices is ~1 GB and
// takes many seconds to write — long enough that an interrupted run leaves a
// truncated, unloadable file. Raw binary is ~260 MB and writes in a fraction
// of a second, and save() below renames a temp file into place so the model
// on disk is never a half-written file.
constexpr char    kBgmdMagic[4] = {'B', 'G', 'M', 'D'};
constexpr int32_t kBgmdVersion  = 1;

void writeMatRaw(std::ofstream& os, const cv::Mat& m) {
    cv::Mat c = m.isContinuous() ? m : m.clone();
    os.write(reinterpret_cast<const char*>(c.data),
             static_cast<std::streamsize>(c.total() * c.elemSize()));
}
} // namespace

// ─── public ──────────────────────────────────────────────────────────────────

cv::Mat BackgroundModel::update(const cv::Mat& image)
{
    CV_Assert(image.type() == CV_8UC1);

    cv::Mat pixels;
    image.convertTo(pixels, CV_32FC1);

    // ── First image: initialise all accumulators ──────────────────────────
    if (image_count_ == 0) {
        cv::Size sz = image.size();
        mean_      = pixels.clone();
        M2_        = cv::Mat::zeros(sz, CV_32FC1);
        count_     = cv::Mat::ones(sz, CV_32FC1);
        dead_mask_ = cv::Mat::zeros(sz, CV_8UC1);
        sat_count_ = cv::Mat::zeros(sz, CV_32FC1);
        zero_count_= cv::Mat::zeros(sz, CV_32FC1);
        ++image_count_;
        return cv::Mat::zeros(sz, CV_8UC1);
    }

    CV_Assert(pixels.size() == mean_.size());

    // ── Accumulate dead-pixel statistics during warmup ────────────────────
    if (image_count_ < dead_pixel_warmup_) {
        cv::Mat sat_px, zero_px;
        cv::threshold(pixels, sat_px,  253.f, 1.f, cv::THRESH_BINARY);
        cv::threshold(pixels, zero_px,   0.f, 1.f, cv::THRESH_BINARY_INV);
        sat_count_  += sat_px;
        zero_count_ += zero_px;
    }

    // ── Build dead mask exactly once at the end of warmup ─────────────────
    if (image_count_ == dead_pixel_warmup_ &&
        !sat_count_.empty() && !zero_count_.empty()) {
        buildDeadMask();
    }

    // ── Compute sigma from M2 (or a floor during early frames) ────────────
    // sigma_ is cached for getSigmaImage(), which used to recompute the whole
    // thing from scratch on every call -- a cv::max, a division and a sqrt over
    // 20.1 Mpixel, each allocating its own 80 MB buffer, on top of what is
    // computed right here. Measured: 59 ms/frame on the M1, ~129 ms on Windows.
    //
    // The two were not computing the same quantity, so they cannot simply be
    // merged:
    //     here            sqrt(max(M2_/count_, 1))   floor on the VARIANCE
    //     getSigmaImage   sqrt(M2_/max(count_, 1))   floor on the COUNT
    // But count_ is initialised to ones() and only ever increments, so
    // max(count_, 1) == count_; and sqrt is monotonic with sqrt(1) == 1, so
    // sqrt(max(v, 1)) == max(sqrt(v), 1). Both therefore derive from one
    // division and one sqrt, and the clamped form is a max on the result.
    // Identical values, half the passes over the image.
    cv::Mat sigma;
    if (image_count_ >= 3) {
        cv::Mat variance = M2_ / count_;
        cv::sqrt(variance, sigma_);                    // = getSigmaImage()'s value
        sigma = cv::max(sigma_, cv::Scalar(1.f));      // = the old clamped value
    } else {
        sigma = cv::Mat::ones(pixels.size(), CV_32FC1) * 5.f;
        sigma_ = cv::Mat();   // getSigmaImage() falls back to deriving it
    }

    // ── Event detection: pixel > mean + n_sigma * sigma  ──────────────────
    cv::Mat threshold_map = mean_ + n_sigma_ * sigma;
    cv::Mat event_mask_f;
    cv::compare(pixels, threshold_map, event_mask_f, cv::CMP_GT);

    // Exclude dead pixels from event mask
    cv::Mat event_mask;
    event_mask_f.copyTo(event_mask);
    if (!dead_mask_.empty()) {
        event_mask.setTo(0, dead_mask_);
    }

    // ── Persistent-event pixels become dead pixels ─────────────────────────
    // Increment the streak where the pixel is an event, reset it elsewhere;
    // pixels reaching DEAD_EVENT_STREAK are hot (real particles never hit
    // the same pixel every frame) and are masked from now on — including
    // from THIS frame's mask, so they stop polluting clusters and the
    // arrival heatmap immediately.
    {
        if (event_streak_.empty())
            event_streak_ = cv::Mat::zeros(pixels.size(), CV_32FC1);  // lazy: resume path
        cv::Mat event01;
        event_mask.convertTo(event01, CV_32FC1, 1.0 / 255.0);
        event_streak_ = (event_streak_ + event01).mul(event01);
        cv::Mat stuck = event_streak_ >= static_cast<float>(DEAD_EVENT_STREAK);
        int n_stuck = cv::countNonZero(stuck);
        if (n_stuck > 0) {
            if (dead_mask_.empty())
                dead_mask_ = cv::Mat::zeros(pixels.size(), CV_8UC1);
            dead_mask_.setTo(255, stuck);
            event_mask.setTo(0, stuck);
            event_streak_.setTo(0.f, stuck);
            // stderr: stdout is a machine-parsed protocol in server mode.
            fprintf(stderr,
                    "[BackgroundModel] %d pixel(s) event for %d consecutive "
                    "frames -> added to dead mask (frame %d)\n",
                    n_stuck, DEAD_EVENT_STREAK, image_count_);
        }
    }

    // ── Welford update on non-event pixels ────────────────────────────────
    cv::Mat normal_mask;
    cv::bitwise_not(event_mask_f, normal_mask);
    if (!dead_mask_.empty()) {
        normal_mask.setTo(0, dead_mask_);
    }
    welfordUpdate(pixels, normal_mask);

    ++image_count_;
    return event_mask;
}

cv::Mat BackgroundModel::getSigmaImage() const
{
    if (M2_.empty()) return cv::Mat();
    // update() already derived exactly this value and cached it. Recomputing
    // here cost three more full passes over 20.1 Mpixel per frame, each with
    // its own 80 MB allocation, for a result that was already in hand.
    if (!sigma_.empty()) return sigma_;
    // Fallback for the first frames (before update() starts caching) and for a
    // model loaded from disk without a subsequent update().
    cv::Mat variance = M2_ / cv::max(count_, cv::Scalar(1.f));
    cv::Mat sigma;
    cv::sqrt(variance, sigma);
    return sigma;
}

void BackgroundModel::save(const std::string& path) const
{
    // Write to a temp file, then atomically rename it over `path`. An
    // interrupted/half-written save leaves only the .tmp behind — the real
    // model on disk is never a truncated, unloadable file.
    const std::string tmp = path + ".tmp";
    {
        std::ofstream os(tmp, std::ios::binary);
        if (!os) throw std::runtime_error("Cannot open model temp file: " + tmp);

        const int32_t rows = mean_.rows, cols = mean_.cols;
        os.write(kBgmdMagic, 4);
        os.write(reinterpret_cast<const char*>(&kBgmdVersion),      sizeof(int32_t));
        os.write(reinterpret_cast<const char*>(&image_count_),      sizeof(int32_t));
        os.write(reinterpret_cast<const char*>(&n_sigma_),          sizeof(float));
        os.write(reinterpret_cast<const char*>(&dead_pixel_warmup_), sizeof(int32_t));
        os.write(reinterpret_cast<const char*>(&rows),              sizeof(int32_t));
        os.write(reinterpret_cast<const char*>(&cols),              sizeof(int32_t));

        writeMatRaw(os, mean_);
        writeMatRaw(os, M2_);
        writeMatRaw(os, count_);
        // dead_mask_ can be empty if update() was never called; write a
        // zero-filled mask of matching size so load() stays symmetric.
        cv::Mat dead = dead_mask_.empty()
                     ? cv::Mat::zeros(mean_.size(), CV_8UC1) : dead_mask_;
        writeMatRaw(os, dead);

        os.flush();
        if (!os) throw std::runtime_error("Failed while writing model: " + tmp);
    } // ofstream closes here, before the rename

    std::filesystem::rename(tmp, path);   // atomic on the same filesystem
}

void BackgroundModel::load(const std::string& path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("Cannot open model file: " + path);

    char    magic[4] = {0, 0, 0, 0};
    int32_t version = 0, rows = 0, cols = 0;
    is.read(magic, 4);
    is.read(reinterpret_cast<char*>(&version), sizeof(int32_t));
    if (std::memcmp(magic, kBgmdMagic, 4) != 0 || version != kBgmdVersion) {
        throw std::runtime_error(
            "Unrecognised or corrupted model file: " + path + " (bad magic/version). "
            "Delete it and rebuild the background model instead of resuming from it.");
    }

    is.read(reinterpret_cast<char*>(&image_count_),       sizeof(int32_t));
    is.read(reinterpret_cast<char*>(&n_sigma_),           sizeof(float));
    is.read(reinterpret_cast<char*>(&dead_pixel_warmup_), sizeof(int32_t));
    is.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
    is.read(reinterpret_cast<char*>(&cols), sizeof(int32_t));
    if (rows <= 0 || cols <= 0)
        throw std::runtime_error("Corrupted model header (rows/cols) in " + path);

    mean_      = cv::Mat(rows, cols, CV_32FC1);
    M2_        = cv::Mat(rows, cols, CV_32FC1);
    count_     = cv::Mat(rows, cols, CV_32FC1);
    dead_mask_ = cv::Mat(rows, cols, CV_8UC1);

    const std::streamsize fbytes = static_cast<std::streamsize>(rows) * cols * sizeof(float);
    const std::streamsize bbytes = static_cast<std::streamsize>(rows) * cols;
    is.read(reinterpret_cast<char*>(mean_.data),      fbytes);
    is.read(reinterpret_cast<char*>(M2_.data),        fbytes);
    is.read(reinterpret_cast<char*>(count_.data),     fbytes);
    is.read(reinterpret_cast<char*>(dead_mask_.data), bbytes);

    // Fail fast on a short/truncated file rather than letting the next
    // update() run `M2_ / count_` on half-filled buffers.
    if (!is || is.gcount() != bbytes) {
        throw std::runtime_error(
            "Corrupted model file: truncated matrix data (" + path + "). "
            "Delete it and rebuild the background model instead of resuming from it.");
    }
}

// ─── private ─────────────────────────────────────────────────────────────────

void BackgroundModel::buildDeadMask()
{
    float warmup = static_cast<float>(dead_pixel_warmup_);

    cv::Mat hot_frac  = sat_count_  / warmup;
    cv::Mat zero_frac = zero_count_ / warmup;

    cv::Mat hot_mask, zero_mask;
    cv::threshold(hot_frac,  hot_mask,  DEAD_SAT_FRACTION,  255, cv::THRESH_BINARY);
    cv::threshold(zero_frac, zero_mask, DEAD_ZERO_FRACTION, 255, cv::THRESH_BINARY);

    cv::Mat combined;
    cv::bitwise_or(hot_mask, zero_mask, combined);
    combined.convertTo(dead_mask_, CV_8UC1);

    int dead_count = cv::countNonZero(dead_mask_);
    sat_count_.release();
    zero_count_.release();

    // stderr, not stdout: in server mode stdout is a machine-parsed protocol
    // and a stray log line there would corrupt a frame's cluster block.
    fprintf(stderr,
            "[BackgroundModel] Dead-pixel mask built: %d dead pixels (%.2f%%) "
            "[n_sigma=%.2f, warmup=%d]\n",
            dead_count,
            100.f * dead_count / static_cast<float>(dead_mask_.total()),
            n_sigma_,
            dead_pixel_warmup_);
}

/**
 * Welford's online algorithm for mean and variance — parallelised row-by-row
 * with OpenMP.  Each row is independent, so no race conditions.
 *
 * Using raw pointer loops instead of OpenCV expression templates lets the
 * compiler vectorise (AVX2) and OpenMP parallelise simultaneously.
 */
void BackgroundModel::welfordUpdate(const cv::Mat& pixels, const cv::Mat& normal_mask)
{
    const int rows = pixels.rows;
    const int cols = pixels.cols;

#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float*   pix_row  = pixels.ptr<float>(r);
        const uint8_t* mask_row = normal_mask.ptr<uint8_t>(r);
        float*         mean_row = mean_.ptr<float>(r);
        float*         M2_row   = M2_.ptr<float>(r);
        float*         cnt_row  = count_.ptr<float>(r);

        for (int c = 0; c < cols; ++c) {
            if (mask_row[c] == 0) continue;   // event or dead pixel — skip

            float x      = pix_row[c];
            float n      = cnt_row[c] + 1.f;
            float delta  = x - mean_row[c];
            float new_mean = mean_row[c] + delta / n;
            float delta2 = x - new_mean;

            cnt_row[c]  = n;
            mean_row[c] = new_mean;
            M2_row[c]  += delta * delta2;
        }
    }
}