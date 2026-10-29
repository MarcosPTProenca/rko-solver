#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <omp.h>

#include "Data.h"

struct HistoryPoint
{
    double elapsed;
    uint64_t evaluations;
    double cost;
};

struct ThreadStats
{
    uint64_t evaluations = 0;
    uint64_t limit = std::numeric_limits<uint64_t>::max();
    bool hit_limit = false;
    double best_cost = std::numeric_limits<double>::infinity();
    std::vector<double> best_keys;
    size_t history_limit = 0;
    uint64_t history_observations = 0;
    std::chrono::steady_clock::time_point started;
    std::vector<HistoryPoint> history;

    void record_history(double elapsed)
    {
        history_observations++;
        history.push_back({elapsed, evaluations, best_cost});
        if (history.size() > history_limit)
        {
            std::vector<HistoryPoint> compacted;
            compacted.reserve(history_limit);
            compacted.push_back(history.front());
            for (size_t k = 2; k + 1 < history.size(); k += 2)
            {
                compacted.push_back(history[k]);
            }
            compacted.push_back(history.back());
            history.swap(compacted);
        }
    }
};

thread_local ThreadStats *t_stats = nullptr;

class StopFlag
{
public:
    bool load() const
    {
        return global_.load() || (t_stats != nullptr && t_stats->hit_limit);
    }
    void store(bool value) { global_.store(value); }

private:
    std::atomic<bool> global_{false};
};

StopFlag stop_execution;

struct EvaluationLimitReached
{
};
thread_local std::mt19937 *t_engine = nullptr;

struct EngineRng
{
    using result_type = std::mt19937::result_type;
    static constexpr result_type min() { return std::mt19937::min(); }
    static constexpr result_type max() { return std::mt19937::max(); }
    result_type operator()() { return (*t_engine)(); }
};

EngineRng rng;
std::vector<TSol> pool;

typedef double (*rko_cpp_eval_fn)(const double *keys, size_t n, void *user);

struct TProblemData
{
    int n;
};

static rko_cpp_eval_fn g_eval = nullptr;
static void *g_user = nullptr;
static std::atomic<int> g_failure(0);

double Decoder(TSol &s, const TProblemData &data)
{
    if (t_stats != nullptr && t_stats->evaluations >= t_stats->limit)
    {
        t_stats->hit_limit = true;
        throw EvaluationLimitReached{};
    }
    double value = g_eval(s.rk.data(), s.rk.size(), g_user);
    if (!std::isfinite(value))
    {
        g_failure.store(1);
        stop_execution.store(true);
        value = 1e300;
    }
    if (t_stats != nullptr)
    {
        t_stats->evaluations++;
        if (t_stats->evaluations >= t_stats->limit)
        {
            t_stats->hit_limit = true;
        }
        if (value < t_stats->best_cost)
        {
            t_stats->best_cost = value;
            t_stats->best_keys.assign(s.rk.begin(), s.rk.end());
            if (t_stats->history_limit > 0)
            {
                t_stats->record_history(
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_stats->started).count());
            }
        }
    }
    return value;
}

#define readParameters rko_vendor_readParameters
#include "Method.h"
#undef readParameters

#define CreateStates rko_vendor_CreateStates
#include "QLearning.h"
#undef CreateStates

struct SpecData
{
    int algorithm;
    bool adaptive;
    std::vector<double> offline;
    std::vector<std::vector<double>> states;
};

thread_local const SpecData *t_spec = nullptr;

void readParameters(const char *method, int control, std::vector<std::vector<double>> &parameters, int numPar)
{
    for (int i = 0; i < numPar; i++)
    {
        parameters[i].push_back(control == 0 ? t_spec->offline[i] : 0.0);
    }
}

void CreateStates(std::vector<std::vector<double>> parameters, int &numStates, int numPar, std::vector<TState> &S)
{
    const std::vector<std::vector<double>> &states = t_spec->states;
    numStates = (int)states.size();
    for (int i = 0; i < numStates; i++)
    {
        TState sAux;
        sAux.label = i;
        sAux.par = states[i];
        sAux.ci = 0;
        sAux.numN = 0;
        sAux.Ai.clear();
        sAux.Qa.clear();
        sAux.sumQ = 0;
        sAux.maxQ = 0;
        sAux.minA = 0;
        sAux.maxA = 0;
        S.push_back(sAux);
    }
    for (int i = 0; i < (int)S.size(); i++)
    {
        for (int j = 0; j < (int)S.size(); j++)
        {
            int distance = 0;
            for (int k = 0; k < (int)S[i].par.size(); k++)
            {
                if (S[i].par[k] != S[j].par[k])
                {
                    distance++;
                }
            }
            if (distance <= 1)
            {
                S[i].Ai.push_back(j);
                double q0 = randomico(0.005, 0.01);
                S[i].Qa.push_back(q0);
                if (q0 > S[i].maxQ)
                {
                    S[i].maxQ = q0;
                    S[i].maxA = S[i].Ai.size() - 1;
                }
            }
        }
    }
}

#include "MH/BRKGA.h"
#include "MH/SA.h"
#include "MH/GRASP.h"
#include "MH/ILS.h"
#include "MH/VNS.h"
#include "MH/PSO.h"
#include "MH/GA.h"
#include "MH/BRKGA_CS.h"
#include "MH/LNS.h"
#include "MH/IPR.h"
#include "MH/MultiStart.h"

extern "C"
{
    struct rko_cpp_spec
    {
        int32_t algorithm;
        int32_t adaptive;
        const double *offline;
        size_t offline_len;
        const double *states;
        size_t state_count;
        size_t state_len;
    };

    struct rko_cpp_config
    {
        double time_limit;
        uint64_t seed;
        int32_t pool_size;
        int32_t strategy;
        uint64_t max_evaluations;
        uint64_t history_limit;
        size_t dimension;
    };

    struct rko_cpp_report
    {
        uint64_t evaluations;
        double elapsed;
        double best_cost;
        int32_t hit_limit;
        uint64_t history_len;
        uint64_t history_observations;
    };

    struct rko_cpp_outputs
    {
        rko_cpp_report *reports;
        double *report_keys;
        rko_cpp_report *initialization;
        double *best_cost;
        double *best_keys;
        double *history_out;
        char *error;
        size_t error_len;
    };
}

static std::mutex g_run_mutex;

static void seed_engine(std::mt19937 &engine, uint64_t seed, uint32_t stream)
{
    std::seed_seq sequence{(uint32_t)(seed & 0xFFFFFFFFu), (uint32_t)(seed >> 32), stream};
    engine.seed(sequence);
    t_engine = &engine;
}

static void run_algorithm(int algorithm, const TRunData &run, const TProblemData &data)
{
    switch (algorithm)
    {
    case 0: SA(run, data); break;
    case 1: ILS(run, data); break;
    case 2: VNS(run, data); break;
    case 3: LNS(run, data); break;
    case 4: BRKGA(run, data); break;
    case 5: BRKGA_CS(run, data); break;
    case 6: PSO(run, data); break;
    case 7: GA(run, data); break;
    case 8: GRASP(run, data); break;
    case 9: IPR(run, data); break;
    default: MultiStart(run, data); break;
    }
}

extern "C" int32_t rko_cpp_run(const rko_cpp_config *config, const rko_cpp_spec *specs, size_t spec_count,
                               rko_cpp_eval_fn eval, void *user, const double *initial_keys,
                               size_t initial_key_count, const rko_cpp_outputs *outputs)
{
    size_t dimension = config->dimension;
    rko_cpp_report *reports = outputs->reports;
    double *report_keys = outputs->report_keys;
    rko_cpp_report *initialization = outputs->initialization;
    double *best_cost = outputs->best_cost;
    double *best_keys = outputs->best_keys;
    double *history_out = outputs->history_out;
    char *error = outputs->error;
    size_t error_len = outputs->error_len;
    std::lock_guard<std::mutex> lock(g_run_mutex);
    if (spec_count == 0 || dimension == 0 || config->time_limit < 0.05 || config->pool_size < 2)
    {
        std::snprintf(error, error_len, "invalid C++ run configuration");
        return 3;
    }
    g_eval = eval;
    g_user = user;
    g_failure.store(0);
    stop_execution.store(false);

    std::vector<SpecData> data_specs(spec_count);
    bool adaptive = false;
    for (size_t i = 0; i < spec_count; i++)
    {
        data_specs[i].algorithm = specs[i].algorithm;
        data_specs[i].adaptive = specs[i].adaptive != 0;
        data_specs[i].offline.assign(specs[i].offline, specs[i].offline + specs[i].offline_len);
        for (size_t s = 0; s < specs[i].state_count; s++)
        {
            data_specs[i].states.emplace_back(specs[i].states + s * specs[i].state_len,
                                              specs[i].states + (s + 1) * specs[i].state_len);
        }
        adaptive = adaptive || data_specs[i].adaptive;
    }

    TProblemData data;
    data.n = (int)dimension;
    TRunData run;
    run.strategy = config->strategy;
    run.control = adaptive ? 1 : 0;
    run.MAXTIME = (int)std::ceil(config->time_limit);
    run.MAXRUNS = 1;
    run.debug = 0;
    run.restart = (float)(config->time_limit / run.MAXTIME);
    run.sizePool = config->pool_size;

    ThreadStats init_stats;
    t_stats = &init_stats;
    std::mt19937 init_engine;
    seed_engine(init_engine, config->seed, 0xFFFFu);
    pool.clear();
    pool.resize(run.sizePool);
    double init_started = get_time_in_seconds();
    CreatePoolSolutions(data, run.sizePool);
    double init_elapsed = get_time_in_seconds() - init_started;
    t_stats = nullptr;
    t_engine = nullptr;
    initialization->evaluations = init_stats.evaluations;
    initialization->elapsed = init_elapsed;
    initialization->best_cost = pool[0].ofv;
    initialization->hit_limit = 0;

    ThreadStats initial_key_stats;
    initial_key_stats.started = std::chrono::steady_clock::now();
    t_stats = &initial_key_stats;
    for (size_t row = 0; row < initial_key_count; row++)
    {
        TSol candidate;
        candidate.rk.assign(initial_keys + row * dimension, initial_keys + (row + 1) * dimension);
        candidate.ofv = Decoder(candidate, data);
        candidate.best_time = get_time_in_seconds();
        std::strcpy(candidate.nameMH, "initial_keys");
        if (g_failure.load() != 0)
        {
            t_stats = nullptr;
            t_engine = nullptr;
            return 1;
        }
        auto duplicate = std::find_if(pool.begin(), pool.end(), [&](const TSol &entry) {
            return entry.rk == candidate.rk;
        });
        if (duplicate != pool.end())
        {
            if (candidate.ofv >= duplicate->ofv)
            {
                continue;
            }
            pool.erase(duplicate);
        }
        if (pool.size() >= (size_t)run.sizePool && candidate.ofv >= pool.back().ofv)
        {
            continue;
        }
        auto position = std::upper_bound(pool.begin(), pool.end(), candidate.ofv,
                                         [](double value, const TSol &entry) { return value < entry.ofv; });
        pool.insert(position, candidate);
        if (pool.size() > (size_t)run.sizePool)
        {
            pool.pop_back();
        }
    }
    t_stats = nullptr;
    t_engine = nullptr;

    std::vector<ThreadStats> stats(spec_count);
    std::vector<double> elapsed(spec_count, 0.0);
    std::vector<std::string> failures(spec_count);
    std::mutex wake_mutex;
    std::condition_variable wake;
    bool finished = false;
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(config->time_limit));
    std::thread watchdog([&]() {
        std::unique_lock<std::mutex> guard(wake_mutex);
        if (!wake.wait_until(guard, deadline, [&]() { return finished; }))
        {
            stop_execution.store(true);
        }
    });

    omp_set_num_threads((int)spec_count);
#pragma omp parallel for schedule(static, 1) num_threads((int)spec_count)
    for (int i = 0; i < (int)spec_count; i++)
    {
        t_spec = &data_specs[i];
        t_stats = &stats[i];
        if (config->max_evaluations > 0)
        {
            t_stats->limit = config->max_evaluations;
        }
        std::mt19937 engine;
        seed_engine(engine, config->seed, (uint32_t)i);
        double started = get_time_in_seconds();
        t_stats->history_limit = (size_t)config->history_limit;
        t_stats->started = std::chrono::steady_clock::now();
        try
        {
            run_algorithm(data_specs[i].algorithm, run, data);
        }
        catch (const EvaluationLimitReached &)
        {
            stats[i].hit_limit = true;
        }
        catch (const std::exception &failure)
        {
            failures[i] = failure.what();
            g_failure.store(2);
            stop_execution.store(true);
        }
        elapsed[i] = get_time_in_seconds() - started;
        t_stats = nullptr;
        t_spec = nullptr;
        t_engine = nullptr;
    }

    {
        std::lock_guard<std::mutex> guard(wake_mutex);
        finished = true;
    }
    wake.notify_all();
    watchdog.join();

    if (g_failure.load() == 1)
    {
        std::snprintf(error, error_len, "objective callbacks must return finite costs");
        return 1;
    }
    if (g_failure.load() == 2)
    {
        for (size_t i = 0; i < spec_count; i++)
        {
            if (!failures[i].empty())
            {
                std::snprintf(error, error_len, "C++ search failed: %s", failures[i].c_str());
                break;
            }
        }
        return 2;
    }

    double overall = std::numeric_limits<double>::infinity();
    std::vector<double> overall_keys;
    if (!pool.empty() && std::isfinite(pool[0].ofv) && pool[0].rk.size() == dimension)
    {
        overall = pool[0].ofv;
        overall_keys = pool[0].rk;
    }
    for (size_t i = 0; i < spec_count; i++)
    {
        reports[i].evaluations = stats[i].evaluations;
        reports[i].elapsed = elapsed[i];
        reports[i].best_cost = stats[i].best_cost;
        reports[i].hit_limit = stats[i].hit_limit ? 1 : 0;
        if (stats[i].history_limit > 0 && std::isfinite(stats[i].best_cost))
        {
            stats[i].record_history(elapsed[i]);
        }
        reports[i].history_len = stats[i].history.size();
        reports[i].history_observations = stats[i].history_observations;
        for (size_t k = 0; k < stats[i].history.size(); k++)
        {
            double *slot = history_out + (i * (config->history_limit + 1) + k) * 3;
            slot[0] = stats[i].history[k].elapsed;
            slot[1] = (double)stats[i].history[k].evaluations;
            slot[2] = stats[i].history[k].cost;
        }
        if (stats[i].best_keys.size() == dimension)
        {
            std::copy(stats[i].best_keys.begin(), stats[i].best_keys.end(), report_keys + i * dimension);
            if (stats[i].best_cost < overall)
            {
                overall = stats[i].best_cost;
                overall_keys = stats[i].best_keys;
            }
        }
    }
    if (overall_keys.size() != dimension)
    {
        std::snprintf(error, error_len, "C++ engine produced no solution");
        return 2;
    }
    *best_cost = overall;
    std::copy(overall_keys.begin(), overall_keys.end(), best_keys);
    return 0;
}
