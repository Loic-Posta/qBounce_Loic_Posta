#pragma once
#include <opencv2/opencv.hpp>
#include <string>

/**
 * @brief Pixel-wise adaptive background model using Welford's online algorithm.
 *
 * For each pixel (x,y) we maintain:
 *   - mean    : running mean of "normal" pixel values
 *   - M2      : running sum of squared deviations (Welford)
 *   - count   : number of samples used
 *
 * A pixel is flagged as an event if:
 *   pixel > mean + n_sigma * sigma
 *
 * Dead/hot pixels (always saturated or always zero) are masked out once
 * after dead_pixel_warmup images and excluded from all subsequent processing.
 *
 * Both n_sigma and dead_pixel_warmup are runtime parameters passed to the
 * constructor (no longer compile-time constants) so they can be set from
 * command-line arguments.
 */
class BackgroundModel {
public:
    // ── compile-time constants (not user-tuneable) ──────────────────────────
    static constexpr float  DEAD_SAT_FRACTION  = 0.95f;  // fraction of warmup where pixel >= 254 → dead
    static constexpr float  DEAD_ZERO_FRACTION = 0.95f;  // fraction of warmup where pixel == 0  → dead
    // A pixel flagged as an EVENT this many frames in a row is a damaged/hot
    // pixel, not physics (real particles never hit the same pixel every
    // frame): it joins the dead mask permanently. Catches hot pixels that
    // are born AFTER the warmup (e.g. radiation damage during a run), which
    // the sat/zero warmup statistics above cannot see.
    static constexpr int    DEAD_EVENT_STREAK  = 5;

    /**
     * @param n_sigma           detection threshold (pixel > mean + n_sigma*sigma)
     * @param dead_pixel_warmup number of images used to build the dead-pixel mask
     */
    explicit BackgroundModel(float n_sigma           = 5.0f,
                             int   dead_pixel_warmup = 18)
        : n_sigma_(n_sigma), dead_pixel_warmup_(dead_pixel_warmup) {}

    /**
     * @brief Feed a new image into the model.
     * @param image   8-bit grayscale image (CV_8UC1)
     * @return        binary event mask (CV_8UC1, 255 = event pixel)
     */
    cv::Mat update(const cv::Mat& image);

    /** Save/load state so you can resume across runs */
    void save(const std::string& path) const;
    void load(const std::string& path);

    /** Access internals for diagnostics */
    cv::Mat getMeanImage()   const { return mean_;  }
    cv::Mat getSigmaImage()  const;
    cv::Mat getDeadMask()    const { return dead_mask_; }
    int     getImageCount()  const { return image_count_; }

private:
    // ── runtime parameters (set once in constructor) ────────────────────────
    float n_sigma_;
    int   dead_pixel_warmup_;

    // ── per-pixel accumulators ──────────────────────────────────────────────
    cv::Mat mean_;          // CV_32FC1
    cv::Mat M2_;            // CV_32FC1  (Welford accumulator)
    cv::Mat count_;         // CV_32FC1  (per-pixel sample count)
    cv::Mat sigma_;         // CV_32FC1  cached by update(); see getSigmaImage()
    cv::Mat dead_mask_;     // CV_8UC1   (255 = dead/hot pixel, exclude)

    // For dead-pixel detection during warmup
    cv::Mat sat_count_;     // how often pixel was >= 254
    cv::Mat zero_count_;    // how often pixel was == 0

    // Consecutive-frames-as-event counter (CV_32FC1) backing DEAD_EVENT_STREAK.
    // Transient: not persisted by save()/load(); the resulting dead_mask_ is.
    cv::Mat event_streak_;

    int image_count_ = 0;

    void buildDeadMask();
    void welfordUpdate(const cv::Mat& pixels, const cv::Mat& normal_mask);
};
