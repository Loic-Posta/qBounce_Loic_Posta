#include "BackgroundModel.h"
#include "ClusterDetector.h"
#include "DataExporter.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <regex>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <stdexcept>
#include <memory>
#include <map>
#include <cstdint>
#include <cstdio>

#ifdef _WIN32
#  include <io.h>
#  include <fcntl.h>
#endif

namespace fs = std::filesystem;

// ─── Configuration ────────────────────────────────────────────────────────────
struct Config {
    std::string  input_folder       = ".";
    std::string  output_csv         = "detections.csv";
    std::string  model_save         = "background_model.yml";
    std::string  export_dir         = "";
    bool         resume             = false;
    bool         save_debug_img     = false;
    std::string  debug_folder       = "debug";
    // Runtime-tuneable model parameters
    float        n_sigma            = 5.0f;
    int          warmup             = 18;
    // Morphological-closing radius (px) applied to the event mask before
    // connected-components labeling, to bridge small gaps in split tracks
    // (e.g. a muon track broken by a couple of under-threshold pixels).
    // 0 = disabled (default; identical behaviour to before this flag existed).
    int          cluster_gap        = 0;
    // Local footprint expansion after clustering: includes weaker neighbouring
    // pixels in a cluster's measured size/charge/bbox without merging distant
    // clusters. 0 = disabled.
    int          cluster_grow       = 0;
    float        cluster_grow_min_snr = 1.0f;
    // Source-level cluster quality floors (0 = disabled, old behaviour).
    // Suppress the noise/gamma/beta cluster flood when only heavy ion
    // tracks matter — see ClusterDetector constructor docs.
    int          min_size_px        = 0;
    float        min_charge_adu     = 0.f;
    // Threading
    int          decode_threads     = 2;   // threads decoding images from disk
    int          detect_threads     = 4;   // threads running cluster characterisation
};

static void printUsage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "  --folder   <path>   Input folder with images          (default: .)\n"
        "  --csv      <path>   Output CSV path                   (default: detections.csv)\n"
        "  --model    <path>   Background model save/load path   (default: background_model.yml)\n"
        "  --export-dir <path> Output directory for analyzed data\n"
        "  --resume            Load existing model and continue\n"
        "  --nsigma   <float>  Detection threshold in sigma      (default: 5.0)\n"
        "  --warmup   <int>    Dead-pixel warmup frame count     (default: 100)\n"
        "  --cluster-gap <int> Morphological-closing radius (px) to bridge\n"
        "                      small gaps in split tracks         (default: 0 = off)\n"
        "  --cluster-grow <int> Local radius (px) to include weak neighbouring\n"
        "                      pixels in each cluster footprint   (default: 0 = off)\n"
        "  --cluster-grow-min-snr <float>\n"
        "                      Minimum SNR for grown neighbour pixels (default: 1.0)\n"
        "  --min-size <int>    Drop clusters smaller than this (px)  (default: 0 = keep all)\n"
        "  --min-charge <float> Drop clusters with total_charge below this ADU\n"
        "                      (default: 0 = keep all)\n"
        "  --dthreads <int>    Image decode thread count         (default: 2)\n"
        "  --cthreads <int>    Cluster detect thread count       (default: 4)\n"
        "  --debug             Save annotated debug images\n"
        "  --debug-folder <p>  Folder for debug images           (default: debug)\n";
}

// ─── Natural sort ─────────────────────────────────────────────────────────────
static bool naturalLess(const fs::path& a, const fs::path& b)
{
    auto tokenise = [](const std::string& s) {
        std::vector<std::pair<bool,std::string>> tokens;
        std::regex re(R"((\d+)|(\D+))");
        for (auto it = std::sregex_iterator(s.begin(), s.end(), re);
             it != std::sregex_iterator(); ++it) {
            tokens.emplace_back((*it)[1].matched, (*it).str());
        }
        return tokens;
    };
    auto ta = tokenise(a.filename().string());
    auto tb = tokenise(b.filename().string());
    for (size_t i = 0; i < std::min(ta.size(), tb.size()); ++i) {
        if (ta[i].first && tb[i].first) {
            int na = std::stoi(ta[i].second), nb = std::stoi(tb[i].second);
            if (na != nb) return na < nb;
        } else {
            if (ta[i].second != tb[i].second) return ta[i].second < tb[i].second;
        }
    }
    return ta.size() < tb.size();
}

// ─── Frame id from filename ─────────────────────────────────────────────────
static int frameIdFromFilenameOrFallback(const fs::path& path, int fallback)
{
    const std::string stem = path.stem().string();
    const std::regex digits_re(R"((\d+))");
    std::smatch match;
    if (std::regex_search(stem, match, digits_re)) {
        return std::stoi(match.str(1));
    }
    return fallback;
}

// ─── Debug visualisation ─────────────────────────────────────────────────────
static void saveDebugImage(const cv::Mat&              raw,
                            const std::vector<Cluster>& clusters,
                            const std::string&          path)
{
    cv::Mat vis;
    cv::cvtColor(raw, vis, cv::COLOR_GRAY2BGR);
    for (const auto& cl : clusters) {
        cv::Scalar color = cl.is_valid ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255);
        cv::rectangle(vis, cl.bounding_box, color, 1);
        cv::circle(vis, {static_cast<int>(cl.center_x),
                         static_cast<int>(cl.center_y)}, 2, color, -1);
    }
    cv::imwrite(path, vis);
}

// ─── Thread-safe CSV writer ───────────────────────────────────────────────────
/**
 * All cluster results are pushed here from worker threads.
 * A dedicated writer thread drains the queue and flushes to disk,
 * so only one thread ever touches the file — no locking on the hot path.
 */
class CsvWriter {
public:
    explicit CsvWriter(const std::string& path) : file_(path) {
        if (!file_) throw std::runtime_error("Cannot open CSV: " + path);
        file_ << Cluster::csvHeader() << "\n";
        writer_ = std::thread([this]{ run(); });
    }

    ~CsvWriter() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            done_ = true;
        }
        cv_.notify_all();
        if (writer_.joinable()) writer_.join();
    }

    /** Called from any thread — enqueues the complete CSV line.
     *  @param is_valid  passed explicitly so the writer thread can count
     *                   without re-parsing the line string.
     */
    void push(std::string line, bool is_valid) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            queue_.push({std::move(line), is_valid});
        }
        cv_.notify_one();
    }

    long long totalEvents() const { return total_events_.load(); }
    long long validEvents() const { return valid_events_.load(); }

private:
    struct Entry { std::string line; bool is_valid; };

    void run() {
        while (true) {
            Entry entry;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this]{ return !queue_.empty() || done_; });
                if (queue_.empty() && done_) break;
                entry = std::move(queue_.front());
                queue_.pop();
            }
            file_ << entry.line << "\n";
            ++total_events_;
            if (entry.is_valid) ++valid_events_;
        }
        file_.flush();
    }

    std::ofstream              file_;
    std::queue<Entry>          queue_;
    std::mutex                 mu_;
    std::condition_variable    cv_;
    bool                       done_ = false;
    std::thread                writer_;
    std::atomic<long long>     total_events_{0};
    std::atomic<long long>     valid_events_{0};
};

// ─── Frame result struct ──────────────────────────────────────────────────────
struct FrameResult {
    int                  frame_id;
    fs::path             path;
    cv::Mat              raw;
    std::vector<Cluster> clusters;
};

// ─── Server mode: persistent process, frames streamed via stdin/stdout ────────
// Protocol per frame:
//   stdin  : [int32 width][int32 height][width*height raw Mono8 bytes]
//   stdout : one CSV line per detected cluster, then "===BATCH_DONE===\n"
// No PNGs, no CSV files — cv::Mat wraps the incoming buffer directly and
// each frame's clusters are written straight back over the pipe.
static void runServerMode(Config& cfg)
{
    // stdin/stdout are a binary/CSV protocol here, not interactive I/O.
    // Two platform pitfalls are handled explicitly:
    //   1. Windows opens stdin/stdout in TEXT mode (CRLF translation) —
    //      that silently corrupts binary frame payloads and appends '\r'
    //      to every protocol line, so force binary mode.
    //   2. Frame payloads are read with fread() against a large stdio
    //      buffer instead of std::cin.read: istream reads chunk through a
    //      small streambuf and, over a pipe, degenerates into thousands of
    //      tiny syscalls (measured ~640 ms per 20 MB frame vs ~30 ms).
#ifdef _WIN32
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    static std::vector<char> stdin_buf(1 << 20);
    setvbuf(stdin, stdin_buf.data(), _IOFBF, stdin_buf.size());
    std::ios::sync_with_stdio(false);   // cout writes bypass C-stdio locking
    std::cin.tie(nullptr);

    std::cerr << "[server] Starting persistent detector (n_sigma=" << cfg.n_sigma
              << "  cluster_gap=" << cfg.cluster_gap
              << "  cluster_grow=" << cfg.cluster_grow << ")\n";

    BackgroundModel bg(cfg.n_sigma, cfg.warmup);
    if (cfg.resume && fs::exists(cfg.model_save)) {
        bg.load(cfg.model_save);
        std::cerr << "[server] Resumed model from " << cfg.model_save
                  << " (frame " << bg.getImageCount() << ")\n";
    }

    ClusterDetector detector(cfg.cluster_gap, cfg.cluster_grow, cfg.cluster_grow_min_snr,
                             cfg.min_size_px, cfg.min_charge_adu);

    // Emit the CSV column header once, before any frame, so the client can
    // name the columns of the header-less per-frame cluster lines that follow.
    // Cluster::csvHeader() stays the single source of truth for column order.
    std::cout << Cluster::csvHeader() << "\n" << std::flush;

    // Arrival-density map accumulated over the whole live run and exported at
    // shutdown: shows WHERE captures happen on the sensor (e.g. a partially
    // covered region), which per-batch counts cannot show.
    cv::Mat live_heatmap;

    // ── Optional data_analysed export (--export-dir) ──────────────────────
    // A dedicated writer thread drains a bounded queue so PNG encoding never
    // sits on the per-frame hot path: the detection loop only enqueues
    // (frame_id, raw, clusters) — a refcounted Mat handle, no pixel copy.
    // If the writer falls behind, frames are skipped from the EXPORT only
    // (counted and reported); detection/counting are never throttled.
    std::unique_ptr<DataExporter> exporter;
    if (!cfg.export_dir.empty())
        exporter = std::make_unique<DataExporter>(fs::current_path(), cfg.export_dir);

    struct ExportJob { int frame_id; cv::Mat raw; std::vector<Cluster> clusters; };
    std::queue<ExportJob>   export_q;
    std::mutex              ex_mu;
    std::condition_variable ex_cv;
    bool                    ex_done    = false;
    int                     ex_dropped = 0;
    int                     ex_written = 0;
    std::thread export_thread;
    if (exporter) {
        export_thread = std::thread([&] {
            while (true) {
                ExportJob job;
                {
                    std::unique_lock<std::mutex> lk(ex_mu);
                    ex_cv.wait(lk, [&] { return !export_q.empty() || ex_done; });
                    if (export_q.empty() && ex_done) break;
                    job = std::move(export_q.front());
                    export_q.pop();
                }
                if (ex_written < DataExporter::RAW_SAMPLE_COUNT) {
                    // No source file exists in server mode — write the raw
                    // frame itself as the reference sample.
                    std::ostringstream nm;
                    nm << "raw_" << std::setw(5) << std::setfill('0')
                       << job.frame_id << ".png";
                    cv::imwrite((exporter->outputDir() / nm.str()).string(), job.raw);
                } else {
                    exporter->exportMaskedFrame(job.frame_id, job.raw, job.clusters);
                }
                ++ex_written;
            }
        });
    }

    int frame_id = static_cast<int>(bg.getImageCount());
    std::vector<char> pixel_buf;
    int32_t header[2];

    while (true) {
        if (std::fread(header, 1, sizeof(header), stdin) != sizeof(header)) {
            std::cerr << "[server] stdin closed, shutting down.\n";
            break;
        }
        const int32_t width = header[0], height = header[1];
        if (width <= 0 || height <= 0) {
            std::cerr << "[server] Invalid header w=" << width << " h=" << height << ", stopping.\n";
            break;
        }

        const size_t n_bytes = static_cast<size_t>(width) * static_cast<size_t>(height);
        pixel_buf.resize(n_bytes);
        auto t_read0 = std::chrono::steady_clock::now();
        const size_t got = std::fread(pixel_buf.data(), 1, n_bytes, stdin);
        auto t_read1 = std::chrono::steady_clock::now();
        if (got != n_bytes) {
            std::cerr << "[server] Short read (" << got << "/" << n_bytes << " bytes), stopping.\n";
            break;
        }
        const double read_ms =
            std::chrono::duration<double, std::milli>(t_read1 - t_read0).count();

        // Clone: pixel_buf is reused next iteration, bg/detector must not
        // hold a dangling view into it.
        cv::Mat raw = cv::Mat(height, width, CV_8UC1, pixel_buf.data()).clone();

        // Per-frame stage timings on stderr: this is the ground truth for
        // "how fast can the live pipeline actually go", and it shows WHICH
        // stage to optimise (bg update vs cluster detection). stdout stays
        // protocol-only.
        auto t0 = std::chrono::steady_clock::now();
        cv::Mat event_mask = bg.update(raw);
        auto t1 = std::chrono::steady_clock::now();
        auto clusters = detector.detect(event_mask, raw, bg.getMeanImage(), bg.getSigmaImage(), frame_id);
        auto t2 = std::chrono::steady_clock::now();

        if (exporter) {
            std::unique_lock<std::mutex> lk(ex_mu);
            if (export_q.size() >= 8) {
                ++ex_dropped;   // export skips a frame; detection never waits
            } else {
                export_q.push({frame_id, raw, clusters});
            }
            lk.unlock();
            ex_cv.notify_one();
        }

        if (live_heatmap.empty())
            live_heatmap = cv::Mat::zeros(raw.size(), CV_32FC1);
        for (const auto& cl : clusters) {
            if (!cl.is_valid) continue;
            for (const auto& p : cl.pixel_coords) {
                if (p.x >= 0 && p.x < live_heatmap.cols &&
                    p.y >= 0 && p.y < live_heatmap.rows)
                    live_heatmap.at<float>(p) += 1.f;
            }
        }

        for (const auto& cl : clusters) {
            std::cout << cl.toCsvLine() << "\n";
        }
        std::cout << "===BATCH_DONE===\n" << std::flush;
        auto t3 = std::chrono::steady_clock::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::cerr << "[server] frame " << frame_id
                  << ": read=" << std::fixed << std::setprecision(0) << read_ms << "ms"
                  << "  update=" << ms(t0, t1) << "ms"
                  << "  detect=" << ms(t1, t2) << "ms"
                  << "  csv_out=" << ms(t2, t3) << "ms"
                  << "  clusters=" << clusters.size() << "\n";

        ++frame_id;
    }

    if (exporter) {
        {
            std::unique_lock<std::mutex> lk(ex_mu);
            ex_done = true;
        }
        ex_cv.notify_all();
        if (export_thread.joinable()) export_thread.join();
        exporter->exportBackgroundModel(bg.getMeanImage(), bg.getSigmaImage());
        if (!live_heatmap.empty())
            exporter->exportSignalHeatmap(live_heatmap);
        std::cerr << "[server] data_analysed export: " << ex_written
                  << " frame(s) written, " << ex_dropped
                  << " skipped (writer backlog) -> " << exporter->outputDir() << "\n";
    }

    if (cfg.resume) {
        bg.save(cfg.model_save);
        std::cerr << "[server] Model saved to " << cfg.model_save << "\n";
    }

    // Export the run's arrival map next to the model file, where the GUI's
    // Results tab looks for it ("Signal arrival (live)" preview).
    if (!live_heatmap.empty() && cv::countNonZero(live_heatmap) > 0) {
        const fs::path dir = fs::path(cfg.model_save).parent_path();
        DataExporter::writeHeatmap(live_heatmap,
                                   dir / "live_signal_arrival.tiff",
                                   dir / "live_signal_arrival_heatmap.png");
        std::cerr << "[server] Live arrival heatmap saved to "
                  << (dir / "live_signal_arrival.tiff") << "\n";
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    Config cfg;
    bool server_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--folder"       && i+1 < argc) cfg.input_folder   = argv[++i];
        else if (arg == "--csv"          && i+1 < argc) cfg.output_csv     = argv[++i];
        else if (arg == "--model"        && i+1 < argc) cfg.model_save     = argv[++i];
        else if (arg == "--export-dir"   && i+1 < argc) cfg.export_dir     = argv[++i];
        else if (arg == "--resume")                     cfg.resume         = true;
        else if (arg == "--debug")                      cfg.save_debug_img = true;
        else if (arg == "--debug-folder" && i+1 < argc) cfg.debug_folder  = argv[++i];
        else if (arg == "--nsigma"       && i+1 < argc) cfg.n_sigma        = std::stof(argv[++i]);
        else if (arg == "--warmup"       && i+1 < argc) cfg.warmup         = std::stoi(argv[++i]);
        else if (arg == "--cluster-gap"  && i+1 < argc) cfg.cluster_gap    = std::stoi(argv[++i]);
        else if (arg == "--cluster-grow" && i+1 < argc) cfg.cluster_grow   = std::stoi(argv[++i]);
        else if (arg == "--cluster-grow-min-snr" && i+1 < argc) cfg.cluster_grow_min_snr = std::stof(argv[++i]);
        else if (arg == "--min-size"     && i+1 < argc) cfg.min_size_px    = std::stoi(argv[++i]);
        else if (arg == "--min-charge"   && i+1 < argc) cfg.min_charge_adu = std::stof(argv[++i]);
        else if (arg == "--dthreads"     && i+1 < argc) cfg.decode_threads = std::stoi(argv[++i]);
        else if (arg == "--cthreads"     && i+1 < argc) cfg.detect_threads = std::stoi(argv[++i]);
        else if (arg == "--server")                     server_mode        = true;
        else if (arg == "--help" || arg == "-h") { printUsage(argv[0]); return 0; }
        else { std::cerr << "[WARN] Unknown argument: " << arg << "\n"; }
    }

    // In server mode stdout is a machine-parsed protocol (CSV header, cluster
    // lines, ===BATCH_DONE===) — any human-readable chatter must go to stderr,
    // otherwise the client mistakes this config line for the CSV header.
    (server_mode ? std::cerr : std::cout)
              << "[main] n_sigma=" << cfg.n_sigma
              << "  warmup=" << cfg.warmup
              << "  cluster_gap=" << cfg.cluster_gap
              << "  cluster_grow=" << cfg.cluster_grow
              << "  cluster_grow_min_snr=" << cfg.cluster_grow_min_snr
              << "  min_size=" << cfg.min_size_px
              << "  min_charge=" << cfg.min_charge_adu
              << "  decode_threads=" << cfg.decode_threads
              << "  detect_threads=" << cfg.detect_threads << "\n";

    if (server_mode) {
        runServerMode(cfg);
        return 0;
    }

    // ── Collect & sort input files ────────────────────────────────────────
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(cfg.input_folder)) {
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".bmp" || ext == ".tiff" || ext == ".tif" || ext == ".png")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), naturalLess);
    std::cout << "[main] Found " << files.size() << " images in " << cfg.input_folder << "\n";
    if (files.empty()) return 0;

    // ── Background model (single-threaded — update() is stateful) ────────
    BackgroundModel bg(cfg.n_sigma, cfg.warmup);
    if (cfg.resume && fs::exists(cfg.model_save)) {
        bg.load(cfg.model_save);
        // Command-line params override saved params when explicitly provided
        // (Model::load() only restores if CLI defaults weren't changed;
        //  here we simply re-construct after load to honour CLI flags.)
        std::cout << "[main] Resumed model from " << cfg.model_save
                  << "  (frame " << bg.getImageCount() << ")\n";
    }

    if (cfg.save_debug_img) fs::create_directories(cfg.debug_folder);

    // ── Thread-safe CSV writer ────────────────────────────────────────────
    CsvWriter csv_writer(cfg.output_csv);

    std::unique_ptr<DataExporter> exporter;
    if (!cfg.export_dir.empty()) {
        exporter = std::make_unique<DataExporter>(cfg.input_folder, cfg.export_dir);
    }

    cv::Mat signal_heatmap;
    std::mutex signal_heatmap_mu;

    // ── Shared queues between stages ──────────────────────────────────────
    //
    //  Pipeline:
    //    [Disk I/O threads] → loaded_queue → [BG model thread] → detect_queue → [Detect threads] → csv_writer
    //
    //  The background model update is inherently sequential (each frame depends on
    //  previous frames), so it runs on the main thread.  Parallelism is exploited:
    //    1. Concurrently decoding the NEXT N images from disk (I/O bound)
    //    2. Running ClusterDetector::detect() on multiple frames in parallel
    //       after the sequential BG step produces their event masks.

    constexpr int QUEUE_CAP = 8;   // bound queues to cap memory (~8 × 20 MP × 4B ≈ 640 MB)

    struct LoadedFrame { int idx; int frame_id; fs::path path; cv::Mat raw; };
    struct DetectJob   { int sequence_idx; int frame_id; fs::path path; cv::Mat raw;
                         cv::Mat event_mask; cv::Mat bg_mean; cv::Mat bg_sigma; };

    // ── Stage 1: parallel image loading ──────────────────────────────────
    std::queue<LoadedFrame>   loaded_q;
    std::mutex                loaded_mu;
    std::condition_variable   loaded_cv;
    std::atomic<int>          next_load_idx{0};

    auto loader_fn = [&]() {
        while (true) {
            int idx = next_load_idx.fetch_add(1);
            if (idx >= static_cast<int>(files.size())) break;
            cv::Mat raw = cv::imread(files[idx].string(), cv::IMREAD_GRAYSCALE);
            if (raw.empty()) {
                std::cerr << "[WARN] Cannot read: " << files[idx] << "\n";
                continue;
            }
            int frame_id = frameIdFromFilenameOrFallback(files[idx], idx);
            {
                std::unique_lock<std::mutex> lk(loaded_mu);
                loaded_cv.wait(lk, [&]{ return static_cast<int>(loaded_q.size()) < QUEUE_CAP; });
                loaded_q.push({idx, frame_id, files[idx], std::move(raw)});
                // Re-sort is not needed: we process in order below; out-of-order
                // arrivals are handled by the reorder buffer.
            }
            loaded_cv.notify_all();
        }
    };

    // Reorder buffer: loaded frames arrive out of order → sort before BG update
    std::map<int, LoadedFrame> reorder_buf;
    std::mutex reorder_mu;

    std::vector<std::thread> loaders;
    for (int t = 0; t < cfg.decode_threads; ++t)
        loaders.emplace_back(loader_fn);

    // ── Stage 2: sequential BG update → feed detect queue ────────────────
    std::queue<DetectJob>    detect_q;
    std::mutex               detect_mu;
    std::condition_variable  detect_cv;
    std::atomic<bool>        detect_done{false};

    // ── Stage 3: parallel cluster detection ──────────────────────────────
    ClusterDetector detector(cfg.cluster_gap, cfg.cluster_grow, cfg.cluster_grow_min_snr,
                             cfg.min_size_px, cfg.min_charge_adu);

    auto detect_fn = [&]() {
        while (true) {
            DetectJob job;
            {
                std::unique_lock<std::mutex> lk(detect_mu);
                detect_cv.wait(lk, [&]{ return !detect_q.empty() || detect_done.load(); });
                if (detect_q.empty() && detect_done.load()) break;
                if (detect_q.empty()) continue;
                job = std::move(detect_q.front());
                detect_q.pop();
            }
            detect_cv.notify_all();

            auto clusters = detector.detect(job.event_mask, job.raw,
                                            job.bg_mean, job.bg_sigma,
                                            job.frame_id);

            // ── Export / compression step ────────────────────────────────
            if (exporter) {
                if (DataExporter::isRawSampleFrame(job.sequence_idx)) {
                    exporter->exportRawSample(job.frame_id, job.path);
                } else {
                    exporter->exportMaskedFrame(job.frame_id, job.raw, clusters);
                }
            }

            if (exporter && cfg.resume) {
                cv::Mat local_heatmap = cv::Mat::zeros(job.raw.size(), CV_32FC1);
                for (const auto& cl : clusters) {
                    if (!cl.is_valid) continue;
                    for (const auto& p : cl.pixel_coords) {
                        if (p.x < 0 || p.x >= local_heatmap.cols ||
                            p.y < 0 || p.y >= local_heatmap.rows) {
                            continue;
                        }
                        local_heatmap.at<float>(p) += 1.0f;
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(signal_heatmap_mu);
                    if (signal_heatmap.empty()) {
                        signal_heatmap = cv::Mat::zeros(job.raw.size(), CV_32FC1);
                    }
                    signal_heatmap += local_heatmap;
                }
            }

            int n_valid = 0;
            for (const auto& cl : clusters) {
                // toCsvLine() now writes all 20 columns including filter_status
                csv_writer.push(cl.toCsvLine(), cl.is_valid);
                if (cl.is_valid) ++n_valid;
            }

            if (!clusters.empty()) {
                std::cout << "[frame " << std::setw(6) << job.frame_id << "]  "
                          << clusters.size() << " clusters  ("
                          << n_valid << " valid)\n";
            }

            if (cfg.save_debug_img && !clusters.empty()) {
                std::string dbg = cfg.debug_folder + "/"
                                + job.path.stem().string() + "_det.png";
                saveDebugImage(job.raw, clusters, dbg);
            }
        }
    };

    std::vector<std::thread> detectors;
    for (int t = 0; t < cfg.detect_threads; ++t)
        detectors.emplace_back(detect_fn);

    // ── Main thread: sequentially drain loaded frames & run BG model ─────
    auto t_start = std::chrono::high_resolution_clock::now();

    // We need frames in order for BG model; use a reorder buffer keyed by idx.
    std::map<int, LoadedFrame> reorder;
    int next_expected = 0;
    int remaining = static_cast<int>(files.size());

    // ── Per-stage accounting for the SERIAL section ──────────────────────
    // Everything below runs on this one thread, so it bounds the whole run no
    // matter how many loader/detector threads are configured -- measured: 2 to
    // 22 threads changes total runtime by ~7%, which is Amdahl's law telling us
    // the answer is here, not in the parallel stages. The clock calls
    // themselves cost nanoseconds against work measured in milliseconds.
    double ms_wait = 0.0;    // blocked waiting for loaders to deliver a frame
    double ms_bg   = 0.0;    // bg.update()  -- the Welford pass
    double ms_clone = 0.0;   // snapshotting mean+sigma for the detect worker
    double ms_mean = 0.0;    //   .. of which: the mean clone
    double ms_sigma = 0.0;   //   .. of which: deriving sigma (max/div/sqrt)
    double ms_push = 0.0;    // blocked pushing into the detect queue
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](clk::time_point t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };

    while (remaining > 0) {
        // Try to pick up newly loaded frames
        {
            auto t0 = clk::now();
            std::unique_lock<std::mutex> lk(loaded_mu);
            loaded_cv.wait_for(lk, std::chrono::milliseconds(5),
                               [&]{ return !loaded_q.empty(); });
            ms_wait += ms_since(t0);
            while (!loaded_q.empty()) {
                auto f = std::move(loaded_q.front());
                loaded_q.pop();
                reorder[f.idx] = std::move(f);
            }
        }
        loaded_cv.notify_all();   // wake loaders that may be blocked on QUEUE_CAP

        // Process all in-order frames available in the reorder buffer
        while (reorder.count(next_expected)) {
            auto& f = reorder[next_expected];

            auto t_bg = clk::now();
            cv::Mat event_mask = bg.update(f.raw);
            ms_bg += ms_since(t_bg);

            // Clone bg state snapshots for the detect worker
            // (bg will continue updating on the next frame immediately)
            DetectJob job;
            job.sequence_idx = f.idx;
            job.frame_id  = f.frame_id;
            job.path      = f.path;
            job.raw       = f.raw;                  // moved into job below
            job.event_mask= event_mask;
            auto t_cl = clk::now();
            // mean_ is live state that bg.update() overwrites on the next
            // frame, and getMeanImage() hands back a shallow cv::Mat header on
            // it -- so the detect worker genuinely needs its own copy.
            job.bg_mean   = bg.getMeanImage().clone();
            ms_mean += ms_since(t_cl);
            auto t_sg = clk::now();
            // sigma is NOT live state: getSigmaImage() derives it on every call
            // (a cv::max, a division and a sqrt over 20.1 Mpixel, each
            // allocating their own 80 MB buffer) and returns a temporary nobody
            // else holds. Cloning it added a fourth 80 MB allocate-and-copy per
            // frame for nothing. Measured cost of the pair before this change:
            // 222.6 ms/frame on Windows, 39.8 ms on the M1.
            // getSigmaImage() no longer returns a private temporary: it hands
            // back the sigma_ that update() caches and OVERWRITES on the next
            // frame. Without this clone the detect worker would be reading a
            // buffer the main thread is rewriting -- a data race that would
            // corrupt results non-deterministically. One 80 MB copy is the
            // price; the saving is the three full-image passes (max, divide,
            // sqrt) that getSigmaImage() used to redo from scratch.
            job.bg_sigma  = bg.getSigmaImage().clone();
            ms_sigma += ms_since(t_sg);
            ms_clone += ms_since(t_cl);

            {
                auto t_pu = clk::now();
                std::unique_lock<std::mutex> lk(detect_mu);
                detect_cv.wait(lk, [&]{ return static_cast<int>(detect_q.size()) < QUEUE_CAP; });
                detect_q.push(std::move(job));
                ms_push += ms_since(t_pu);
            }
            detect_cv.notify_all();

            reorder.erase(next_expected);
            ++next_expected;
            --remaining;
        }
    }

    // ── Shutdown ──────────────────────────────────────────────────────────
    for (auto& t : loaders)   t.join();

    {
        std::unique_lock<std::mutex> lk(detect_mu);
        detect_done = true;
    }
    detect_cv.notify_all();
    for (auto& t : detectors) t.join();

    // CsvWriter destructor flushes and joins its writer thread

    bg.save(cfg.model_save);
    std::cout << "[main] Model saved to " << cfg.model_save << "\n";

    // ── Export final background-model state (mean/sigma), once ─────────────
    if (exporter) {
        exporter->exportBackgroundModel(bg.getMeanImage(), bg.getSigmaImage());
        std::cout << "[main] Background model images exported to "
                  << exporter->outputDir() << "\n";
        if (cfg.resume && !signal_heatmap.empty()) {
            exporter->exportSignalHeatmap(signal_heatmap);
            std::cout << "[main] Signal arrival heatmap exported to "
                      << exporter->outputDir() << "\n";
        }
    }

    auto t_end   = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n=== Summary ===\n"
              << "  Frames processed : " << bg.getImageCount()          << "\n"
              << "  Total clusters   : " << csv_writer.totalEvents()    << "\n"
              << "  Valid clusters   : " << csv_writer.validEvents()    << "\n"
              << "  Elapsed          : " << std::fixed << std::setprecision(1)
                                          << elapsed << " s\n"
              << "  Throughput       : "
              << std::setprecision(2)
              << static_cast<double>(bg.getImageCount()) / elapsed
              << " frames/s\n";

    // Where the SERIAL thread spent its time. Loading and cluster detection run
    // on their own threads and overlap with this, so they do not appear here --
    // that is the point: this section is the floor the whole run cannot beat,
    // and it is what any optimisation has to attack.
    {
        const int n = std::max(1, bg.getImageCount());
        const double serial = ms_wait + ms_bg + ms_clone + ms_push;
        auto row = [&](const char* name, double ms) {
            std::cout << "    " << std::left << std::setw(22) << name
                      << std::right << std::setw(8) << std::setprecision(1)
                      << ms / 1000.0 << " s"
                      << std::setw(9) << std::setprecision(1) << ms / n << " ms/img"
                      << std::setw(7) << std::setprecision(0)
                      << (serial > 0 ? 100.0 * ms / serial : 0.0) << " %\n";
        };
        std::cout << std::fixed
                  << "  Serial-stage profile (this thread bounds the run):\n";
        row("waiting for loaders", ms_wait);
        row("bg.update (Welford)", ms_bg);
        row("snapshot mean+sigma", ms_clone);
        row("  .. mean clone",     ms_mean);
        row("  .. sigma derive",   ms_sigma);
        row("push to detect queue", ms_push);
        std::cout << "    " << std::left << std::setw(22) << "TOTAL serial"
                  << std::right << std::setw(8) << std::setprecision(1)
                  << serial / 1000.0 << " s"
                  << std::setw(9) << std::setprecision(1) << serial / n << " ms/img"
                  << std::setw(7) << std::setprecision(0)
                  << (elapsed > 0 ? 100.0 * serial / (elapsed * 1000.0) : 0.0)
                  << " % of wall\n";
    }

    return 0;
}
