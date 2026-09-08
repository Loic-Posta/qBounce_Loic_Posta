#include "DataExporter.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

static void drawLabel(cv::Mat& canvas, const std::string& text, const cv::Point& org)
{
    cv::putText(canvas, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(30, 30, 30), 3, cv::LINE_AA);
    cv::putText(canvas, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(250, 250, 250), 1, cv::LINE_AA);
}

// Writes `src` losslessly (no rescaling/quantisation) so exact ADU-scale
// values round-trip bit-for-bit — used both for the background model
// (mean/sigma) and, now, the signal arrival heatmap, so every heatmap the
// Python dashboard displays has a raw numeric counterpart to plot from
// (rather than having to reverse-engineer values out of the PNG preview).
static void saveRawTiff(const cv::Mat& src, const fs::path& dest)
{
    if (src.empty()) {
        std::cerr << "[DataExporter] WARNING: empty matrix — skipping raw export for "
                  << dest << "\n";
        return;
    }
    if (!cv::imwrite(dest.string(), src)) {
        std::cerr << "[DataExporter] WARNING: failed to write " << dest << "\n";
    }
}

static void savePlasmaHeatmap(const cv::Mat& src, const fs::path& dest,
                              bool allow_log_scale = true)
{
    if (src.empty()) return;

    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(src, &min_val, &max_val);

    bool use_log_scale = false;
    if (allow_log_scale && max_val > 0.0) {
        // Use the *true* smallest positive value in the matrix for the
        // variance-ratio test, rather than assuming 1.0 whenever min_val
        // isn't positive. Sparse data (e.g. mostly-zero signal counts)
        // often has min_val == 0, and silently substituting 1.0 there can
        // understate the true max/min ratio and fail to trigger log scale
        // when the data actually warrants it.
        double positive_min = min_val;
        if (positive_min <= 0.0) {
            cv::Mat positive_mask = src > 0.0;
            if (cv::countNonZero(positive_mask) > 0) {
                double true_pos_min = 0.0, unused_max = 0.0;
                cv::minMaxLoc(src, &true_pos_min, &unused_max, nullptr, nullptr, positive_mask);
                positive_min = true_pos_min;
            } else {
                positive_min = 1.0;  // no positive values at all — nothing to scale
            }
        }
        use_log_scale = (max_val / positive_min) >= 100.0;
    }

    cv::Mat processed;
    src.convertTo(processed, CV_32FC1);
    if (use_log_scale) {
        processed += 1.0f;
        cv::log(processed, processed);
    }

    cv::Mat normalized;
    if (max_val - min_val < 1e-12) {
        normalized = cv::Mat::zeros(src.size(), CV_8UC1);
    } else {
        cv::normalize(processed, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    }

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_PLASMA);

    const int gap = 12;
    const int bar_w = 26;
    const int legend_w = 110;

    cv::Mat colorbar(normalized.rows, bar_w, CV_8UC1);
    for (int y = 0; y < colorbar.rows; ++y) {
        float ratio = 1.0f - static_cast<float>(y) / std::max(colorbar.rows - 1, 1);
        uint8_t value = static_cast<uint8_t>(std::round(ratio * 255.0f));
        colorbar.row(y).setTo(value);
    }

    cv::Mat colorbar_bgr;
    cv::applyColorMap(colorbar, colorbar_bgr, cv::COLORMAP_PLASMA);

    cv::Mat legend(normalized.rows, legend_w, CV_8UC3, cv::Scalar(245, 245, 245));
    std::ostringstream min_ss;
    std::ostringstream max_ss;
    min_ss << std::fixed << std::setprecision(3) << min_val;
    max_ss << std::fixed << std::setprecision(3) << max_val;

    drawLabel(legend, use_log_scale ? "log1p" : "linear", cv::Point(8, 24));
    drawLabel(legend, max_ss.str(), cv::Point(8, 52));
    drawLabel(legend, min_ss.str(), cv::Point(8, normalized.rows - 16));

    cv::Mat canvas(normalized.rows,
                   colored.cols + gap + colorbar_bgr.cols + gap + legend.cols,
                   CV_8UC3, cv::Scalar(255, 255, 255));
    colored.copyTo(canvas(cv::Rect(0, 0, colored.cols, colored.rows)));
    colorbar_bgr.copyTo(canvas(cv::Rect(colored.cols + gap, 0,
                                        colorbar_bgr.cols, colorbar_bgr.rows)));
    legend.copyTo(canvas(cv::Rect(colored.cols + gap + colorbar_bgr.cols + gap,
                                  0, legend.cols, legend.rows)));

    if (!cv::imwrite(dest.string(), canvas)) {
        std::cerr << "[DataExporter] WARNING: failed to write heatmap preview "
                  << dest << "\n";
    }
}

// ─── construction ───────────────────────────────────────────────────────────

DataExporter::DataExporter(const fs::path& input_folder, const fs::path& output_dir)
{
    fs::path abs_input = fs::absolute(input_folder);
    if (abs_input.filename().empty())          // strip a trailing "/" if present
        abs_input = abs_input.parent_path();

    if (!output_dir.empty()) {
        output_dir_ = fs::absolute(output_dir);
    } else {
        const fs::path data_dir = abs_input.parent_path();                // e.g. .../Data
        const fs::path analysed_root = data_dir.parent_path() / "data_analysed"; // sibling of Data
        output_dir_ = analysed_root / abs_input.filename();               // .../data_analysed/dark
    }

    std::error_code ec;
    fs::create_directories(output_dir_, ec);
    if (ec) {
        throw std::runtime_error(
            "DataExporter: could not create output directory " +
            output_dir_.string() + ": " + ec.message());
    }
    // stderr, never stdout: in --server mode stdout is a strict machine
    // protocol (CSV header, per-frame cluster rows, ===BATCH_DONE===) read
    // line-by-line by the Python engine. This constructor runs mid-stream
    // when --export-dir is used live, so a stdout print here lands inside
    // whatever frame is being read at that instant — the engine reads it as
    // a garbled cluster row (all physics columns NaN), which ONNX still
    // "classifies" into a meaningless label. Offline runs still show this
    // line: PipelineController merges child stderr into the Console via
    // subprocess.STDOUT.
    std::cerr << "[DataExporter] Exporting to " << output_dir_ << "\n";
}

// ─── raw samples (first frames in processing order) ─────────────────────────

void DataExporter::exportRawSample(int frame_id, const fs::path& source_path) const
{
    std::ostringstream name;
    name << "raw_" << std::setw(5) << std::setfill('0') << frame_id
         << source_path.extension().string();
    const fs::path dest = output_dir_ / name.str();

    std::error_code ec;
    fs::copy_file(source_path, dest, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        std::cerr << "[DataExporter] WARNING: could not copy raw sample "
                  << source_path << " -> " << dest << ": " << ec.message() << "\n";
    }
}

// ─── sparse masked frames ──────────────────────────────────────────────────

void DataExporter::exportMaskedFrame(int frame_id, const cv::Mat& raw,
                                      const std::vector<Cluster>& clusters) const
{
    CV_Assert(raw.type() == CV_8UC1);

    // Start fully black; only paint in pixels that belong to a valid cluster.
    cv::Mat masked = cv::Mat::zeros(raw.size(), CV_8UC1);

    for (const auto& cl : clusters) {
        if (!cl.is_valid) continue;   // rejected clusters stay black
        for (const auto& p : cl.pixel_coords) {
            if (p.x < 0 || p.x >= raw.cols || p.y < 0 || p.y >= raw.rows)
                continue;   // defensive; shouldn't happen, coords come from this same frame
            masked.at<uint8_t>(p) = raw.at<uint8_t>(p);
        }
    }

    std::ostringstream name;
    name << "masked_" << std::setw(5) << std::setfill('0') << frame_id << ".png";
    const fs::path dest = output_dir_ / name.str();

    // Maximum lossless PNG compression — ideal for mostly-black sparse images.
    const std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 9};
    if (!cv::imwrite(dest.string(), masked, params)) {
        std::cerr << "[DataExporter] WARNING: failed to write " << dest << "\n";
    }
}

// ─── final background model state ─────────────────────────────────────────

void DataExporter::exportBackgroundModel(const cv::Mat& mean, const cv::Mat& sigma) const
{
    // mean_/sigma are CV_32FC1. Written straight to 32-bit float TIFF: no
    // rescaling or quantisation, so the exact ADU-scale values the model
    // computed round-trip bit-for-bit (verified: imwrite/imread CV_32FC1
    // .tiff is lossless in OpenCV's TIFF codec).
    const fs::path mean_path  = output_dir_ / "background_mean.tiff";
    const fs::path sigma_path = output_dir_ / "background_sigma.tiff";

    if (mean.empty() || sigma.empty()) {
        std::cerr << "[DataExporter] WARNING: mean/sigma image is empty — "
                     "skipping background model export.\n";
        return;
    }

    saveRawTiff(mean, mean_path);
    saveRawTiff(sigma, sigma_path);

    savePlasmaHeatmap(mean, output_dir_ / "background_mean_heatmap.png", false);
    savePlasmaHeatmap(sigma, output_dir_ / "background_sigma_heatmap.png", true);
}

void DataExporter::exportSignalHeatmap(const cv::Mat& counts) const
{
    if (counts.empty()) {
        std::cerr << "[DataExporter] WARNING: signal heatmap is empty — skipping preview.\n";
        return;
    }
    saveRawTiff(counts, output_dir_ / "signal_arrival.tiff");
    savePlasmaHeatmap(counts, output_dir_ / "signal_arrival_heatmap.png", true);
}

void DataExporter::writeHeatmap(const cv::Mat& counts,
                                const fs::path& tiff_path,
                                const fs::path& png_path)
{
    if (counts.empty()) {
        std::cerr << "[DataExporter] WARNING: heatmap is empty — skipping "
                  << tiff_path << "\n";
        return;
    }
    saveRawTiff(counts, tiff_path);
    savePlasmaHeatmap(counts, png_path, true);
}
