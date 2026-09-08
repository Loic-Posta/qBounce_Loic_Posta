#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>

#include "ClusterDetector.h"   // for Cluster

/**
 * @brief Writes a compressed, storage-friendly summary of one processed
 *        input folder to a configurable output directory.
 *
 * If no explicit output directory is provided, output goes to
 * ".../data_analysed/<subfolder>" — i.e. data_analysed is created next to
 * the "Data" directory, and mirrors whichever subfolder (dark / signal /
 * ...) is currently being run.
 *
 * Per subdirectory processed:
 *   - first RAW_SAMPLE_COUNT frames   : byte-for-byte copy of the source
 *                                       image file (exportRawSample)
 *   - remaining frames                : sparse masked PNG — only pixels
 *                                       belonging to a *valid* cluster keep
 *                                       their raw value, everything else is
 *                                       0 (exportMaskedFrame)
 *   - once, after the last frame      : final background mean/sigma as
 *                                       lossless 32-bit float TIFFs plus
 *                                       plasma-colored PNG previews
 *                                       (exportBackgroundModel)
 *   - once, after signal processing   : plasma-colored arrival heatmap PNG
 *                                       (exportSignalHeatmap)
 *
 * Thread-safety: exportRawSample() / exportMaskedFrame() may be called
 * concurrently from multiple detect-worker threads — each call only touches
 * a file named after its own frame_id, so there's no shared mutable state.
 * exportBackgroundModel() should be called once, after all worker threads
 * have joined and the model has processed its last frame.
 */
class DataExporter {
public:
    static constexpr int RAW_SAMPLE_COUNT = 5;

    /**
     * Creates the output directory (and any missing parents) immediately.
     * @param input_folder  the folder actually being processed, e.g. ".../Data/dark"
     * @throws std::runtime_error if the output directory cannot be created.
     */
    explicit DataExporter(const std::filesystem::path& input_folder,
                          const std::filesystem::path& output_dir = {});

    /** True for processing-order indices that should be copied as raw samples. */
    static bool isRawSampleFrame(int sequence_idx) { return sequence_idx < RAW_SAMPLE_COUNT; }

    /** Copy the original file byte-for-byte, naming it with the real frame_id. */
    void exportRawSample(int frame_id, const std::filesystem::path& source_path) const;

    /**
     * Write a sparse masked PNG (max compression), named with the real frame_id,
     * where only pixels belonging to clusters with is_valid == true keep
     * their raw value; everything else (background + rejected clusters) is 0.
     * @param raw       CV_8UC1 source frame (unmodified)
     * @param clusters  clusters detected in this frame (as returned by ClusterDetector::detect)
     */
    void exportMaskedFrame(int frame_id, const cv::Mat& raw,
                            const std::vector<Cluster>& clusters) const;

    /** Call once, after the pipeline has finished, with the model's final mean/sigma. */
    void exportBackgroundModel(const cv::Mat& mean, const cv::Mat& sigma) const;

    /** Call once after signal detection to export the arrival-density heatmap preview. */
    void exportSignalHeatmap(const cv::Mat& counts) const;

    /**
     * Standalone heatmap writer (raw float TIFF + plasma PNG preview with
     * auto log scale) for callers that have no exporter instance — e.g. the
     * live server exporting its end-of-run arrival map next to the model
     * file instead of into a data_analysed/ tree.
     */
    static void writeHeatmap(const cv::Mat& counts,
                             const std::filesystem::path& tiff_path,
                             const std::filesystem::path& png_path);

    const std::filesystem::path& outputDir() const { return output_dir_; }

private:
    std::filesystem::path output_dir_;
};
