// Japan Live Derivatives Pricing and Risk Framework - C++17 conversion
// --------------------------------------------------------------------
// Converted from the Python notebook/package workflow.  The calculation
// logic mirrors the Python implementation: Black-Scholes for equities,
// Black-76 for futures, implied volatility, binomial tree, Monte Carlo,
// finite-difference Greeks, scenario analysis, trading signals, and
// threshold optimization from 1% to 10%.
//
// External libraries are intentionally avoided to keep the framework easy
// to build in a quant/research environment. Yahoo Finance access is handled
// through an isolated curl-based adapter. For production trading, replace
// that adapter with a broker/exchange/internal market-data feed.

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace quant {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double PI = 3.141592653589793238462643383279502884;

// ---------------------------------------------------------------------
// Generic utilities
// ---------------------------------------------------------------------

std::string ltrim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return s;
}

std::string rtrim(std::string s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
    return s;
}

std::string trim(std::string s) { return rtrim(ltrim(std::move(s))); }

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool isFinite(double x) { return std::isfinite(x); }

std::string stripQuotes(std::string s) {
    s = trim(std::move(s));
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

std::string normalizeOptionType(const std::string& value) {
    const std::string raw = upper(trim(value));
    if (raw == "CALL" || raw == "C" || raw == "CE") return "CALL";
    if (raw == "PUT" || raw == "P" || raw == "PE") return "PUT";
    throw std::runtime_error("Invalid option type: " + value);
}

double toDouble(const std::string& s, double defaultValue = NaN) {
    try {
        std::string cleaned;
        cleaned.reserve(s.size());
        for (char ch : s) {
            if (ch != ',') cleaned.push_back(ch);
        }
        cleaned = trim(cleaned);
        if (cleaned.empty() || lower(cleaned) == "nan" || lower(cleaned) == "null") {
            return defaultValue;
        }
        size_t idx = 0;
        const double v = std::stod(cleaned, &idx);
        return idx > 0 ? v : defaultValue;
    } catch (...) {
        return defaultValue;
    }
}

int toInt(const std::string& s, int defaultValue = 0) {
    try {
        return std::stoi(trim(s));
    } catch (...) {
        return defaultValue;
    }
}

double roundN(double x, int decimals) {
    if (!isFinite(x)) return x;
    const double scale = std::pow(10.0, decimals);
    return std::round(x * scale) / scale;
}

std::string formatDouble(double x, int decimals) {
    if (!isFinite(x)) return "";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals) << x;
    return oss.str();
}

std::string safeName(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    bool lastUnderscore = false;
    for (unsigned char ch : name) {
        if (std::isalnum(ch)) {
            out.push_back(static_cast<char>(ch));
            lastUnderscore = false;
        } else if (!lastUnderscore) {
            out.push_back('_');
            lastUnderscore = true;
        }
    }
    while (!out.empty() && out.front() == '_') out.erase(out.begin());
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out;
}

std::string escapeCsv(const std::string& s) {
    const bool needsQuotes = s.find_first_of(",\n\r\"") != std::string::npos;
    if (!needsQuotes) return s;
    std::string out = "\"";
    for (char ch : s) {
        if (ch == '"') out += "\"\"";
        else out.push_back(ch);
    }
    out += "\"";
    return out;
}

std::vector<std::string> parseCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"');
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (ch == ',' && !inQuotes) {
            fields.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    fields.push_back(cur);
    return fields;
}

std::map<std::string, size_t> headerIndex(const std::vector<std::string>& header) {
    std::map<std::string, size_t> idx;
    for (size_t i = 0; i < header.size(); ++i) {
        idx[lower(trim(header[i]))] = i;
    }
    return idx;
}

std::string fieldAt(const std::vector<std::string>& row, const std::map<std::string, size_t>& idx,
                    const std::string& name, const std::string& defaultValue = "") {
    auto it = idx.find(lower(name));
    if (it == idx.end() || it->second >= row.size()) return defaultValue;
    return trim(row[it->second]);
}

void ensureDirs(const std::vector<std::string>& paths) {
    for (const auto& path : paths) {
        if (!path.empty()) fs::create_directories(path);
    }
}

// ---------------------------------------------------------------------
// Date handling
// ---------------------------------------------------------------------

struct Date {
    int y = 1970;
    int m = 1;
    int d = 1;

    std::string iso() const {
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << y << "-"
            << std::setw(2) << std::setfill('0') << m << "-"
            << std::setw(2) << std::setfill('0') << d;
        return oss.str();
    }

    std::string yyyymmdd() const {
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << y
            << std::setw(2) << std::setfill('0') << m
            << std::setw(2) << std::setfill('0') << d;
        return oss.str();
    }
};

bool operator<(const Date& a, const Date& b) {
    return std::tie(a.y, a.m, a.d) < std::tie(b.y, b.m, b.d);
}

bool operator>=(const Date& a, const Date& b) { return !(a < b); }

Date parseDate(const std::string& raw) {
    std::string s = trim(raw);
    if (s.size() >= 10 && s[4] == '-' && s[7] == '-') {
        return {toInt(s.substr(0, 4)), toInt(s.substr(5, 2)), toInt(s.substr(8, 2))};
    }
    std::string digits;
    for (char ch : s) {
        if (std::isdigit(static_cast<unsigned char>(ch))) digits.push_back(ch);
    }
    if (digits.size() < 8) throw std::runtime_error("Invalid date: " + raw);
    return {toInt(digits.substr(0, 4)), toInt(digits.substr(4, 2)), toInt(digits.substr(6, 2))};
}

std::time_t dateToTimeT(const Date& date) {
    std::tm tm{};
    tm.tm_year = date.y - 1900;
    tm.tm_mon = date.m - 1;
    tm.tm_mday = date.d;
    tm.tm_hour = 0;
    tm.tm_min = 0;
    tm.tm_sec = 0;
    tm.tm_isdst = -1;
    return std::mktime(&tm);
}

long daysBetween(const Date& from, const Date& to) {
    const double seconds = std::difftime(dateToTimeT(to), dateToTimeT(from));
    return static_cast<long>(std::floor(seconds / 86400.0 + 0.5));
}

Date todayLocal() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    return {tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday};
}

std::string epochToDate(long long epochSeconds) {
    const std::time_t tt = static_cast<std::time_t>(epochSeconds);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d");
    return oss.str();
}

std::string epochToDateTime(long long epochSeconds) {
    const std::time_t tt = static_cast<std::time_t>(epochSeconds);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S UTC");
    return oss.str();
}

// ---------------------------------------------------------------------
// Domain structs
// ---------------------------------------------------------------------

struct Instrument {
    std::string id;
    std::string name;
    std::string symbol;
    std::string assetType;
    std::string yahoo;
    std::string tradingview;
    std::string optionType = "CALL";
    double dividendYield = 0.0;
};

struct Config {
    std::string projectName = "Japan Live Derivatives Pricing and Risk Framework";
    std::string dataDir = "data";
    std::string outputDir = "outputs";
    std::string plotDir = "outputs/plots";

    std::string startDate = "2024-01-01";
    int tradingDays = 252;
    double riskFreeRate = 0.01;
    double mispricingThreshold = 0.05;

    int mcPaths = 10000;
    int mcSteps = 252;
    int mcSeed = 42;

    std::string optionChainCsv = "data/option_chain_japan.csv";
    std::string quoteSource = "YFINANCE";
    std::vector<Instrument> instruments;
};

struct OptionRow {
    std::string instrumentId;
    Date expiry;
    double strike = NaN;
    std::string optionType;
    double marketPrice = NaN;
    double volume = NaN;
    double openInterest = NaN;
    double multiplier = 1.0;
};

struct PricePoint {
    std::string date;
    double close = NaN;
};

struct ReturnStats {
    double historicalVol = NaN;
    double annualizedReturn = NaN;
    double skewness = NaN;
    double kurtosis = NaN;
};

struct Greeks {
    double delta = NaN;
    double gamma = NaN;
    double vega = NaN;
    double theta = NaN;
    double rho = NaN;
};

struct McResult {
    double price = NaN;
    double error = NaN;
};

struct Signal {
    double edgePct = NaN;
    std::string signal;
    std::string action;
    std::string hedgeAction;
    std::string reason;
};

struct ScenarioRow {
    std::string scenario;
    double spotOrFuture = NaN;
    double volatility = NaN;
    double riskFreeRate = NaN;
    double optionPrice = NaN;
};

struct SummaryRow {
    std::string instrument;
    std::string symbol;
    std::string assetType;
    std::string yahoo;
    std::string tradingview;
    std::string quoteTimestamp;
    double spotOrFuture = NaN;
    std::string expiry;
    double strike = NaN;
    std::string optionType;
    double marketPrice = NaN;
    double historicalVol = NaN;
    double impliedVol = NaN;
    double volUsed = NaN;
    double fairValue = NaN;
    double binomial = NaN;
    double monteCarlo = NaN;
    double mcError = NaN;
    double delta = NaN;
    double gamma = NaN;
    double vega = NaN;
    double theta = NaN;
    double rho = NaN;
    double annualizedReturn = NaN;
    double skewness = NaN;
    double kurtosis = NaN;
};

struct ThresholdSignalRow {
    double threshold = NaN;
    double thresholdPercent = NaN;
    std::string instrument;
    double marketPrice = NaN;
    double fairValue = NaN;
    double edgePct = NaN;
    double edgePercent = NaN;
    std::string signal;
    std::string action;
    std::string hedgeAction;
    std::string reason;
};

struct ThresholdSummaryRow {
    double threshold = NaN;
    double thresholdPercent = NaN;
    int tradeSignals = 0;
    int buySignals = 0;
    int sellSignals = 0;
    int holdSignals = 0;
    double avgAbsEdgePercent = 0.0;
    double totalAbsEdgePercent = 0.0;
    double optimizationScore = 0.0;
};

// ---------------------------------------------------------------------
// Minimal YAML parser for the project config shape used by the notebook
// ---------------------------------------------------------------------

void assignInstrumentField(Instrument& inst, const std::string& key, const std::string& value) {
    const std::string k = lower(key);
    if (k == "id") inst.id = value;
    else if (k == "name") inst.name = value;
    else if (k == "symbol") inst.symbol = value;
    else if (k == "asset_type") inst.assetType = lower(value);
    else if (k == "yahoo") inst.yahoo = value;
    else if (k == "tradingview") inst.tradingview = value;
    else if (k == "option_type") inst.optionType = normalizeOptionType(value);
    else if (k == "dividend_yield") inst.dividendYield = toDouble(value, 0.0);
}

std::pair<std::string, std::string> parseYamlKeyValue(const std::string& line) {
    const size_t pos = line.find(':');
    if (pos == std::string::npos) return {"", ""};
    const std::string key = trim(line.substr(0, pos));
    const std::string value = stripQuotes(line.substr(pos + 1));
    return {key, value};
}

Config loadConfig(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open config file: " + path);

    Config cfg;
    std::string section;
    Instrument current;
    bool inInstrument = false;

    auto flushInstrument = [&]() {
        if (inInstrument && !current.id.empty()) {
            if (current.assetType.empty()) current.assetType = "equity";
            if (current.optionType.empty()) current.optionType = "CALL";
            cfg.instruments.push_back(current);
        }
        current = Instrument{};
        inInstrument = false;
    };

    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        if (trim(line).empty()) continue;

        const std::string stripped = trim(line);
        if (line[0] != ' ' && stripped.back() == ':') {
            if (section == "instruments") flushInstrument();
            section = stripped.substr(0, stripped.size() - 1);
            continue;
        }

        if (section == "instruments") {
            std::string item = ltrim(line);
            if (item.rfind("- ", 0) == 0) {
                flushInstrument();
                inInstrument = true;
                item = trim(item.substr(2));
                if (!item.empty()) {
                    auto [key, value] = parseYamlKeyValue(item);
                    assignInstrumentField(current, key, value);
                }
            } else if (inInstrument) {
                auto [key, value] = parseYamlKeyValue(item);
                assignInstrumentField(current, key, value);
            }
            continue;
        }

        auto [key, value] = parseYamlKeyValue(stripped);
        if (key.empty()) continue;

        if (section == "project") {
            if (key == "name") cfg.projectName = value;
            else if (key == "data_dir") cfg.dataDir = value;
            else if (key == "output_dir") cfg.outputDir = value;
            else if (key == "plot_dir") cfg.plotDir = value;
        } else if (section == "market") {
            if (key == "start_date") cfg.startDate = value;
            else if (key == "trading_days") cfg.tradingDays = toInt(value, 252);
            else if (key == "risk_free_rate") cfg.riskFreeRate = toDouble(value, 0.01);
            else if (key == "mispricing_threshold") cfg.mispricingThreshold = toDouble(value, 0.05);
        } else if (section == "monte_carlo") {
            if (key == "paths") cfg.mcPaths = toInt(value, 10000);
            else if (key == "steps") cfg.mcSteps = toInt(value, 252);
            else if (key == "seed") cfg.mcSeed = toInt(value, 42);
        } else if (section == "data") {
            if (key == "option_chain_csv") cfg.optionChainCsv = value;
            else if (key == "quote_source") cfg.quoteSource = value;
        }
    }
    if (section == "instruments") flushInstrument();

    if (cfg.instruments.empty()) throw std::runtime_error("No instruments found in config.");
    return cfg;
}

// ---------------------------------------------------------------------
// Market and option-chain data loading
// ---------------------------------------------------------------------

std::vector<OptionRow> loadOptionChain(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open option-chain CSV: " + path);

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Option-chain CSV is empty: " + path);
    auto idx = headerIndex(parseCsvLine(line));

    for (const auto& col : {"instrument_id", "expiry", "strike", "option_type", "market_price"}) {
        if (idx.find(col) == idx.end()) {
            throw std::runtime_error("Missing required option-chain column: " + std::string(col));
        }
    }

    std::vector<OptionRow> rows;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        const auto fields = parseCsvLine(line);
        OptionRow row;
        row.instrumentId = upper(trim(fieldAt(fields, idx, "instrument_id")));
        row.expiry = parseDate(fieldAt(fields, idx, "expiry"));
        row.strike = toDouble(fieldAt(fields, idx, "strike"));
        row.optionType = normalizeOptionType(fieldAt(fields, idx, "option_type"));
        row.marketPrice = toDouble(fieldAt(fields, idx, "market_price"));
        row.volume = toDouble(fieldAt(fields, idx, "volume"));
        row.openInterest = toDouble(fieldAt(fields, idx, "open_interest"));
        row.multiplier = toDouble(fieldAt(fields, idx, "multiplier"), 1.0);

        if (isFinite(row.strike) && row.strike > 0.0 && isFinite(row.marketPrice) && row.marketPrice > 0.0) {
            rows.push_back(row);
        }
    }
    return rows;
}

std::vector<PricePoint> loadHistoricalCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open historical CSV: " + path);

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("Historical CSV is empty: " + path);
    auto idx = headerIndex(parseCsvLine(line));
    if (idx.find("date") == idx.end() || idx.find("close") == idx.end()) {
        throw std::runtime_error("Historical CSV must contain date,close columns: " + path);
    }

    std::vector<PricePoint> out;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        const auto fields = parseCsvLine(line);
        PricePoint p;
        p.date = fieldAt(fields, idx, "date");
        p.close = toDouble(fieldAt(fields, idx, "close"));
        if (!p.date.empty() && isFinite(p.close) && p.close > 0.0) out.push_back(p);
    }
    return out;
}

std::string urlEncode(const std::string& s) {
    std::ostringstream oss;
    oss << std::uppercase << std::hex;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            oss << static_cast<char>(c);
        } else {
            oss << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(c);
        }
    }
    return oss.str();
}

std::optional<std::string> runCurl(const std::string& url) {
#if defined(_WIN32)
    (void)url;
    return std::nullopt;
#else
    const std::string command = "curl -L --silent --max-time 20 '" + url + "' 2>/dev/null";
    std::array<char, 4096> buffer{};
    std::string result;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return std::nullopt;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }
    const int rc = pclose(pipe);
    if (rc != 0 || result.empty()) return std::nullopt;
    return result;
#endif
}

std::optional<std::vector<double>> parseDoubleArrayByJsonKey(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = json.find(needle, pos)) != std::string::npos) {
        const size_t colon = json.find(':', pos + needle.size());
        const size_t open = json.find('[', colon == std::string::npos ? pos : colon);
        if (open == std::string::npos) return std::nullopt;
        size_t cursor = open + 1;
        while (cursor < json.size() && std::isspace(static_cast<unsigned char>(json[cursor]))) ++cursor;
        if (cursor < json.size() && json[cursor] == '{') {
            pos = cursor + 1;
            continue; // outer array containing an object; seek the inner numeric array
        }
        const size_t close = json.find(']', open);
        if (close == std::string::npos) return std::nullopt;
        const std::string body = json.substr(open + 1, close - open - 1);
        std::vector<double> values;
        std::stringstream ss(body);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim(token);
            if (token.empty() || token == "null") values.push_back(NaN);
            else values.push_back(toDouble(token));
        }
        return values;
    }
    return std::nullopt;
}

std::optional<std::vector<long long>> parseLongArrayByJsonKey(const std::string& json, const std::string& key) {
    auto doubles = parseDoubleArrayByJsonKey(json, key);
    if (!doubles) return std::nullopt;
    std::vector<long long> out;
    out.reserve(doubles->size());
    for (double v : *doubles) {
        if (isFinite(v)) out.push_back(static_cast<long long>(v));
        else out.push_back(0);
    }
    return out;
}

std::optional<std::vector<PricePoint>> fetchYahooHistory(const std::string& yahooSymbol, const std::string& startDate) {
    const Date start = parseDate(startDate);
    const auto period1 = static_cast<long long>(dateToTimeT(start));
    const auto now = std::chrono::system_clock::now();
    const auto period2 = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    const std::string url = "https://query1.finance.yahoo.com/v8/finance/chart/" +
        urlEncode(yahooSymbol) + "?period1=" + std::to_string(period1) +
        "&period2=" + std::to_string(period2) +
        "&interval=1d&events=history&includeAdjustedClose=true";

    auto json = runCurl(url);
    if (!json || json->find("timestamp") == std::string::npos) return std::nullopt;

    auto timestamps = parseLongArrayByJsonKey(*json, "timestamp");
    auto closes = parseDoubleArrayByJsonKey(*json, "adjclose");
    if (!closes) closes = parseDoubleArrayByJsonKey(*json, "close");
    if (!timestamps || !closes) return std::nullopt;

    const size_t n = std::min(timestamps->size(), closes->size());
    std::vector<PricePoint> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if ((*timestamps)[i] > 0 && isFinite((*closes)[i]) && (*closes)[i] > 0.0) {
            out.push_back({epochToDate((*timestamps)[i]), (*closes)[i]});
        }
    }
    if (out.empty()) return std::nullopt;
    return out;
}

std::optional<std::pair<double, std::string>> fetchLatestYahooQuote(const std::string& yahooSymbol) {
    const std::string url = "https://query1.finance.yahoo.com/v8/finance/chart/" +
        urlEncode(yahooSymbol) + "?range=5d&interval=5m&includePrePost=false";

    auto json = runCurl(url);
    if (!json || json->find("timestamp") == std::string::npos) return std::nullopt;

    auto timestamps = parseLongArrayByJsonKey(*json, "timestamp");
    auto closes = parseDoubleArrayByJsonKey(*json, "close");
    if (!timestamps || !closes) return std::nullopt;

    const size_t n = std::min(timestamps->size(), closes->size());
    for (size_t i = n; i > 0; --i) {
        const size_t idx = i - 1;
        if ((*timestamps)[idx] > 0 && isFinite((*closes)[idx]) && (*closes)[idx] > 0.0) {
            return std::make_pair((*closes)[idx], epochToDateTime((*timestamps)[idx]));
        }
    }
    return std::nullopt;
}

ReturnStats calculateReturnStats(const std::vector<PricePoint>& prices, int tradingDays) {
    std::vector<double> returns;
    for (size_t i = 1; i < prices.size(); ++i) {
        if (prices[i - 1].close > 0.0 && prices[i].close > 0.0) {
            returns.push_back(std::log(prices[i].close / prices[i - 1].close));
        }
    }
    if (returns.size() < 2) throw std::runtime_error("Not enough data to calculate volatility.");

    const int n = static_cast<int>(returns.size());
    const double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / n;
    double sq = 0.0;
    for (double x : returns) sq += (x - mean) * (x - mean);
    const double sampleStd = std::sqrt(sq / (n - 1));

    ReturnStats stats;
    stats.historicalVol = sampleStd * std::sqrt(static_cast<double>(tradingDays));
    stats.annualizedReturn = mean * tradingDays;

    if (sampleStd > 0.0 && n > 2) {
        double s3 = 0.0;
        for (double x : returns) {
            const double z = (x - mean) / sampleStd;
            s3 += z * z * z;
        }
        // Matches pandas/scipy bias-corrected skew convention closely.
        stats.skewness = (static_cast<double>(n) / ((n - 1.0) * (n - 2.0))) * s3;
    }

    if (sampleStd > 0.0 && n > 3) {
        double s4 = 0.0;
        for (double x : returns) {
            const double z = (x - mean) / sampleStd;
            s4 += z * z * z * z;
        }
        // Fisher excess kurtosis with small-sample correction, as pandas reports.
        stats.kurtosis = (static_cast<double>(n) * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * s4
                       - (3.0 * (n - 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0)));
    }
    return stats;
}

std::optional<OptionRow> selectAtmOption(const std::vector<OptionRow>& chain, const Instrument& instrument,
                                         double spotOrFuture, const Date& tradeDate) {
    std::vector<OptionRow> candidates;
    const std::string id = upper(instrument.id);
    const std::string optType = normalizeOptionType(instrument.optionType);
    for (const auto& row : chain) {
        if (row.instrumentId == id && row.expiry >= tradeDate && row.optionType == optType) {
            candidates.push_back(row);
        }
    }
    if (candidates.empty()) return std::nullopt;

    const auto nearestExpiry = std::min_element(candidates.begin(), candidates.end(), [](const OptionRow& a, const OptionRow& b) {
        return a.expiry < b.expiry;
    })->expiry;

    std::optional<OptionRow> best;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (const auto& row : candidates) {
        if (row.expiry.y == nearestExpiry.y && row.expiry.m == nearestExpiry.m && row.expiry.d == nearestExpiry.d) {
            const double dist = std::abs(row.strike - spotOrFuture);
            if (dist < bestDistance) {
                bestDistance = dist;
                best = row;
            }
        }
    }
    return best;
}

// ---------------------------------------------------------------------
// Pricing model functions
// ---------------------------------------------------------------------

double normCdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double blackScholesPrice(double S, double K, double T, double r, double sigma,
                         const std::string& optionType, double q = 0.0) {
    const std::string type = normalizeOptionType(optionType);
    sigma = std::max(sigma, 1e-8);

    if (T <= 0.0) return type == "CALL" ? std::max(S - K, 0.0) : std::max(K - S, 0.0);

    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    const double d2 = d1 - sigma * std::sqrt(T);

    if (type == "CALL") {
        return S * std::exp(-q * T) * normCdf(d1) - K * std::exp(-r * T) * normCdf(d2);
    }
    return K * std::exp(-r * T) * normCdf(-d2) - S * std::exp(-q * T) * normCdf(-d1);
}

double black76Price(double F, double K, double T, double r, double sigma,
                    const std::string& optionType) {
    const std::string type = normalizeOptionType(optionType);
    sigma = std::max(sigma, 1e-8);

    if (T <= 0.0) return type == "CALL" ? std::max(F - K, 0.0) : std::max(K - F, 0.0);

    const double d1 = (std::log(F / K) + 0.5 * sigma * sigma * T) / (sigma * std::sqrt(T));
    const double d2 = d1 - sigma * std::sqrt(T);
    const double discount = std::exp(-r * T);

    if (type == "CALL") return discount * (F * normCdf(d1) - K * normCdf(d2));
    return discount * (K * normCdf(-d2) - F * normCdf(-d1));
}

double modelPrice(double S_or_F, double K, double T, double r, double sigma,
                  const std::string& optionType, const std::string& assetType, double q = 0.0) {
    if (lower(assetType) == "futures") return black76Price(S_or_F, K, T, r, sigma, optionType);
    return blackScholesPrice(S_or_F, K, T, r, sigma, optionType, q);
}

double impliedVolatility(double marketPrice, double S_or_F, double K, double T, double r,
                         const std::string& optionType, const std::string& assetType, double q = 0.0) {
    if (marketPrice <= 0.0 || T <= 0.0 || S_or_F <= 0.0 || K <= 0.0) return NaN;

    auto objective = [&](double sigma) {
        return modelPrice(S_or_F, K, T, r, sigma, optionType, assetType, q) - marketPrice;
    };

    double lo = 0.0001;
    double hi = 5.0;
    double fLo = objective(lo);
    double fHi = objective(hi);
    if (!isFinite(fLo) || !isFinite(fHi) || fLo * fHi > 0.0) return NaN;

    double mid = 0.5 * (lo + hi);
    for (int i = 0; i < 200; ++i) {
        mid = 0.5 * (lo + hi);
        const double fMid = objective(mid);
        if (std::abs(fMid) < 1e-10 || (hi - lo) < 1e-10) return mid;
        if (fLo * fMid <= 0.0) {
            hi = mid;
            fHi = fMid;
        } else {
            lo = mid;
            fLo = fMid;
        }
        (void)fHi;
    }
    return mid;
}

double binomialTreePrice(double S_or_F, double K, double T, double r, double sigma,
                         int steps, const std::string& optionType, const std::string& assetType,
                         double q = 0.0) {
    const std::string type = normalizeOptionType(optionType);
    sigma = std::max(sigma, 1e-8);
    steps = std::max(1, steps);

    if (T <= 0.0) return type == "CALL" ? std::max(S_or_F - K, 0.0) : std::max(K - S_or_F, 0.0);

    const double dt = T / steps;
    const double u = std::exp(sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double carry = lower(assetType) == "futures" ? 0.0 : r - q;
    double p = (std::exp(carry * dt) - d) / (u - d);
    p = std::clamp(p, 0.0, 1.0);

    std::vector<double> values(steps + 1);
    for (int j = 0; j <= steps; ++j) {
        const double price = S_or_F * std::pow(u, j) * std::pow(d, steps - j);
        values[j] = type == "CALL" ? std::max(price - K, 0.0) : std::max(K - price, 0.0);
    }

    const double discount = std::exp(-r * dt);
    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            values[j] = discount * (p * values[j + 1] + (1.0 - p) * values[j]);
        }
    }
    return values[0];
}

McResult monteCarloPrice(double S_or_F, double K, double T, double r, double sigma,
                         const std::string& optionType, const std::string& assetType, double q,
                         int paths, int steps, int seed) {
    const std::string type = normalizeOptionType(optionType);
    sigma = std::max(sigma, 1e-8);
    paths = std::max(1, paths);
    steps = std::max(1, steps);

    if (T <= 0.0) {
        const double value = type == "CALL" ? std::max(S_or_F - K, 0.0) : std::max(K - S_or_F, 0.0);
        return {value, 0.0};
    }

    std::mt19937_64 rng(static_cast<unsigned long long>(seed));
    std::normal_distribution<double> normal(0.0, 1.0);

    const double dt = T / steps;
    const double drift = lower(assetType) == "futures" ? -0.5 * sigma * sigma : r - q - 0.5 * sigma * sigma;
    const double volStep = sigma * std::sqrt(dt);
    const double discount = std::exp(-r * T);

    std::vector<double> discounted;
    discounted.reserve(static_cast<size_t>(paths));

    for (int p = 0; p < paths; ++p) {
        double logReturn = 0.0;
        for (int step = 0; step < steps; ++step) {
            logReturn += drift * dt + volStep * normal(rng);
        }
        const double terminal = S_or_F * std::exp(logReturn);
        const double payoff = type == "CALL" ? std::max(terminal - K, 0.0) : std::max(K - terminal, 0.0);
        discounted.push_back(discount * payoff);
    }

    const double mean = std::accumulate(discounted.begin(), discounted.end(), 0.0) / discounted.size();
    double sq = 0.0;
    for (double x : discounted) sq += (x - mean) * (x - mean);
    const double stdDev = discounted.size() > 1 ? std::sqrt(sq / (discounted.size() - 1)) : 0.0;
    return {mean, stdDev / std::sqrt(static_cast<double>(paths))};
}

Greeks finiteDifferenceGreeks(double S_or_F, double K, double T, double r, double sigma,
                              const std::string& optionType, const std::string& assetType,
                              double q = 0.0) {
    const double hS = std::max(S_or_F * 0.001, 0.01);
    const double hSig = 0.01;
    const double hR = 0.0001;
    const double hT = 1.0 / 365.0;

    const double p0 = modelPrice(S_or_F, K, T, r, sigma, optionType, assetType, q);
    const double pUp = modelPrice(S_or_F + hS, K, T, r, sigma, optionType, assetType, q);
    const double pDn = modelPrice(std::max(S_or_F - hS, 0.01), K, T, r, sigma, optionType, assetType, q);

    Greeks g;
    g.delta = (pUp - pDn) / (2.0 * hS);
    g.gamma = (pUp - 2.0 * p0 + pDn) / (hS * hS);
    g.vega = modelPrice(S_or_F, K, T, r, sigma + hSig, optionType, assetType, q) - p0;
    g.theta = modelPrice(S_or_F, K, std::max(T - hT, 1.0 / 365.0), r, sigma, optionType, assetType, q) - p0;
    g.rho = (modelPrice(S_or_F, K, T, r + hR, sigma, optionType, assetType, q) - p0) / hR;
    return g;
}

// ---------------------------------------------------------------------
// Risk, signal, and threshold logic
// ---------------------------------------------------------------------

std::vector<ScenarioRow> scenarioAnalysis(double S_or_F, double K, double T, double r, double sigma,
                                          const std::string& optionType, const std::string& assetType,
                                          double q) {
    struct ScenarioSpec { std::string name; double spotMult; double volMult; double rateShift; };
    const std::vector<ScenarioSpec> specs = {
        {"Low Volatility", 1.00, 0.70,  0.0000},
        {"High Volatility", 1.00, 1.50,  0.0000},
        {"Bull Market",     1.10, 1.00,  0.0000},
        {"Bear Market",     0.90, 1.20,  0.0000},
        {"Market Crash",    0.75, 2.00, -0.0025},
    };

    std::vector<ScenarioRow> rows;
    for (const auto& s : specs) {
        const double scenarioSpot = S_or_F * s.spotMult;
        const double scenarioSigma = std::max(sigma * s.volMult, 0.0001);
        const double scenarioR = std::max(r + s.rateShift, 0.0);
        rows.push_back({s.name, scenarioSpot, scenarioSigma, scenarioR,
                        modelPrice(scenarioSpot, K, T, scenarioR, scenarioSigma, optionType, assetType, q)});
    }
    return rows;
}

Signal generateSignal(double marketPrice, double fairValue, double /* delta */, double threshold) {
    if (!isFinite(marketPrice) || marketPrice <= 0.0) {
        return {NaN, "NO_MARKET_PRICE", "NONE", "NONE", "Real option market price unavailable."};
    }

    const double edgePct = (fairValue - marketPrice) / marketPrice;
    if (edgePct > threshold) {
        return {edgePct, "BUY_OPTION_DELTA_HEDGE", "BUY", "SELL", "Option is cheaper than model fair value."};
    }
    if (edgePct < -threshold) {
        return {edgePct, "SELL_OPTION_DELTA_HEDGE", "SELL", "BUY", "Option is expensive compared with model fair value."};
    }
    return {edgePct, "HOLD_NO_EDGE", "NONE", "NONE", "No strong mispricing edge."};
}

std::pair<std::vector<ThresholdSummaryRow>, std::vector<ThresholdSignalRow>>
runThresholdOptimization(const std::vector<SummaryRow>& summaryRows) {
    std::vector<ThresholdSummaryRow> thresholdSummary;
    std::vector<ThresholdSignalRow> allSignals;

    for (int i = 1; i <= 10; ++i) {
        const double threshold = i / 100.0;
        std::vector<ThresholdSignalRow> tempRows;

        for (const auto& row : summaryRows) {
            const double edgePct = (row.fairValue - row.marketPrice) / row.marketPrice;
            std::string signal;
            std::string action;
            std::string hedgeAction;
            std::string reason;

            if (edgePct > threshold) {
                signal = "BUY_OPTION_DELTA_HEDGE";
                action = "BUY";
                hedgeAction = "SELL";
                reason = "Option is cheaper than model fair value.";
            } else if (edgePct < -threshold) {
                signal = "SELL_OPTION_DELTA_HEDGE";
                action = "SELL";
                hedgeAction = "BUY";
                reason = "Option is expensive compared with model fair value.";
            } else {
                signal = "HOLD_NO_EDGE";
                action = "NONE";
                hedgeAction = "NONE";
                reason = "No strong mispricing edge.";
            }

            tempRows.push_back({threshold, threshold * 100.0, row.instrument, row.marketPrice,
                                row.fairValue, edgePct, edgePct * 100.0, signal, action,
                                hedgeAction, reason});
        }

        int tradeCount = 0;
        int buyCount = 0;
        int sellCount = 0;
        int holdCount = 0;
        double totalAbsEdge = 0.0;
        for (const auto& r : tempRows) {
            if (r.action != "NONE") {
                ++tradeCount;
                totalAbsEdge += std::abs(r.edgePercent);
            }
            if (r.action == "BUY") ++buyCount;
            else if (r.action == "SELL") ++sellCount;
            else ++holdCount;
        }

        const double avgAbsEdge = tradeCount > 0 ? totalAbsEdge / tradeCount : 0.0;
        const double optimizationScore = tradeCount > 0 ? avgAbsEdge * std::sqrt(static_cast<double>(tradeCount)) : 0.0;
        thresholdSummary.push_back({threshold, threshold * 100.0, tradeCount, buyCount, sellCount,
                                    holdCount, avgAbsEdge, totalAbsEdge, optimizationScore});

        allSignals.insert(allSignals.end(), tempRows.begin(), tempRows.end());
    }

    return {thresholdSummary, allSignals};
}

// ---------------------------------------------------------------------
// Output generation
// ---------------------------------------------------------------------

void writePricingSummaryCsv(const std::string& path, const std::vector<SummaryRow>& rows) {
    std::ofstream out(path);
    out << "instrument,symbol,asset_type,yahoo,tradingview,quote_timestamp,spot_or_future,expiry,strike,option_type,market_price,"
           "historical_vol,implied_vol,vol_used,fair_value,binomial,monte_carlo,mc_error,delta,gamma,vega,theta,rho,"
           "annualized_return,skewness,kurtosis\n";
    for (const auto& r : rows) {
        out << escapeCsv(r.instrument) << ',' << escapeCsv(r.symbol) << ',' << r.assetType << ',' << r.yahoo << ','
            << escapeCsv(r.tradingview) << ',' << escapeCsv(r.quoteTimestamp) << ','
            << formatDouble(r.spotOrFuture, 4) << ',' << r.expiry << ',' << formatDouble(r.strike, 4) << ',' << r.optionType << ','
            << formatDouble(r.marketPrice, 4) << ',' << formatDouble(r.historicalVol, 6) << ',' << formatDouble(r.impliedVol, 6) << ','
            << formatDouble(r.volUsed, 6) << ',' << formatDouble(r.fairValue, 4) << ',' << formatDouble(r.binomial, 4) << ','
            << formatDouble(r.monteCarlo, 4) << ',' << formatDouble(r.mcError, 6) << ',' << formatDouble(r.delta, 6) << ','
            << formatDouble(r.gamma, 8) << ',' << formatDouble(r.vega, 6) << ',' << formatDouble(r.theta, 6) << ','
            << formatDouble(r.rho, 6) << ',' << formatDouble(r.annualizedReturn, 6) << ',' << formatDouble(r.skewness, 6) << ','
            << formatDouble(r.kurtosis, 6) << '\n';
    }
}

void writeTradingSignalsCsv(const std::string& path, const std::vector<std::pair<std::string, Signal>>& rows) {
    std::ofstream out(path);
    out << "instrument,signal,action,hedge_action,edge_pct,reason\n";
    for (const auto& [instrument, s] : rows) {
        out << escapeCsv(instrument) << ',' << s.signal << ',' << s.action << ',' << s.hedgeAction << ','
            << formatDouble(roundN(s.edgePct, 6), 6) << ',' << escapeCsv(s.reason) << '\n';
    }
}

void writeScenarioCsv(const std::string& path, const std::vector<ScenarioRow>& rows) {
    std::ofstream out(path);
    out << "scenario,spot_or_future,volatility,risk_free_rate,option_price\n";
    for (const auto& r : rows) {
        out << escapeCsv(r.scenario) << ',' << formatDouble(r.spotOrFuture, 6) << ',' << formatDouble(r.volatility, 6)
            << ',' << formatDouble(r.riskFreeRate, 6) << ',' << formatDouble(r.optionPrice, 6) << '\n';
    }
}

void writeThresholdSummaryCsv(const std::string& path, const std::vector<ThresholdSummaryRow>& rows) {
    std::ofstream out(path);
    out << "threshold,threshold_%,trade_signals,buy_signals,sell_signals,hold_signals,avg_abs_edge_%,total_abs_edge_%,optimization_score\n";
    for (const auto& r : rows) {
        out << formatDouble(r.threshold, 2) << ',' << formatDouble(r.thresholdPercent, 2) << ','
            << r.tradeSignals << ',' << r.buySignals << ',' << r.sellSignals << ',' << r.holdSignals << ','
            << formatDouble(r.avgAbsEdgePercent, 6) << ',' << formatDouble(r.totalAbsEdgePercent, 6) << ','
            << formatDouble(r.optimizationScore, 6) << '\n';
    }
}

void writeAllThresholdSignalsCsv(const std::string& path, const std::vector<ThresholdSignalRow>& rows) {
    std::ofstream out(path);
    out << "threshold,threshold_%,instrument,market_price,fair_value,edge_pct,edge_%,signal,action,hedge_action,reason\n";
    for (const auto& r : rows) {
        out << formatDouble(r.threshold, 2) << ',' << formatDouble(r.thresholdPercent, 2) << ',' << escapeCsv(r.instrument) << ','
            << formatDouble(r.marketPrice, 4) << ',' << formatDouble(r.fairValue, 4) << ',' << formatDouble(r.edgePct, 10) << ','
            << formatDouble(r.edgePercent, 6) << ',' << r.signal << ',' << r.action << ',' << r.hedgeAction << ','
            << escapeCsv(r.reason) << '\n';
    }
}

std::string markdownTable(const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& rows) {
    std::ostringstream out;
    out << "|";
    for (const auto& h : headers) out << ' ' << h << " |";
    out << "\n|";
    for (size_t i = 0; i < headers.size(); ++i) out << "---|";
    out << "\n";
    for (const auto& row : rows) {
        out << "|";
        for (const auto& cell : row) out << ' ' << cell << " |";
        out << "\n";
    }
    return out.str();
}

void generateReport(const std::string& path, const std::vector<SummaryRow>& summary,
                    const std::vector<std::pair<std::string, Signal>>& signals,
                    const std::map<std::string, std::vector<ScenarioRow>>& scenarioTables,
                    const std::vector<ThresholdSummaryRow>& thresholdSummary) {
    std::ofstream out(path);
    out << "# Japan Advanced Derivatives Pricing and Risk Modeling Framework\n\n";
    out << "## Strategy Used\n\n";
    out << "The strategy compares real option market price with model fair value. "
           "If the option is cheaper than fair value, the model gives BUY_OPTION_DELTA_HEDGE. "
           "If the option is expensive compared with fair value, the model gives SELL_OPTION_DELTA_HEDGE. "
           "Otherwise, the model gives HOLD_NO_EDGE.\n\n";

    out << "## Pricing Summary\n\n";
    if (summary.empty()) {
        out << "No pricing rows generated.\n\n";
    } else {
        std::vector<std::vector<std::string>> rows;
        for (const auto& r : summary) {
            rows.push_back({r.instrument, r.assetType, formatDouble(r.spotOrFuture, 4), formatDouble(r.strike, 4),
                            formatDouble(r.marketPrice, 4), formatDouble(r.fairValue, 4), formatDouble(r.binomial, 4),
                            formatDouble(r.monteCarlo, 4), formatDouble(r.delta, 6)});
        }
        out << markdownTable({"instrument", "asset_type", "spot_or_future", "strike", "market_price", "fair_value", "binomial", "monte_carlo", "delta"}, rows) << "\n";
    }

    out << "## Trading Signals\n\n";
    if (signals.empty()) {
        out << "No trading signals generated.\n\n";
    } else {
        std::vector<std::vector<std::string>> rows;
        for (const auto& [inst, sig] : signals) {
            rows.push_back({inst, sig.signal, sig.action, sig.hedgeAction, formatDouble(sig.edgePct, 6), sig.reason});
        }
        out << markdownTable({"instrument", "signal", "action", "hedge_action", "edge_pct", "reason"}, rows) << "\n";
    }

    out << "## Threshold Optimization Summary\n\n";
    if (!thresholdSummary.empty()) {
        std::vector<std::vector<std::string>> rows;
        for (const auto& r : thresholdSummary) {
            rows.push_back({formatDouble(r.thresholdPercent, 2), std::to_string(r.tradeSignals), std::to_string(r.buySignals),
                            std::to_string(r.sellSignals), std::to_string(r.holdSignals), formatDouble(r.avgAbsEdgePercent, 4),
                            formatDouble(r.optimizationScore, 4)});
        }
        out << markdownTable({"threshold_%", "trade_signals", "buy", "sell", "hold", "avg_abs_edge_%", "optimization_score"}, rows) << "\n";
    }

    out << "## Scenario Analysis\n\n";
    for (const auto& [name, rows] : scenarioTables) {
        out << "### " << name << "\n\n";
        std::vector<std::vector<std::string>> mdRows;
        for (const auto& r : rows) {
            mdRows.push_back({r.scenario, formatDouble(r.spotOrFuture, 4), formatDouble(r.volatility, 6),
                              formatDouble(r.riskFreeRate, 6), formatDouble(r.optionPrice, 4)});
        }
        out << markdownTable({"scenario", "spot_or_future", "volatility", "risk_free_rate", "option_price"}, mdRows) << "\n";
    }

    out << "## Important Notes\n\n";
    out << "- This C++ version preserves the Python pricing, risk, signal, and threshold-optimization logic.\n";
    out << "- Yahoo/curl market data is a research adapter; production trading should use a broker/exchange-grade feed.\n";
    out << "- Option-chain prices in CSV must be replaced with real market data before final use.\n";
    out << "- This model does not guarantee profit.\n";
}

// Simple SVG chart helpers. These replace the Python Matplotlib PNGs with
// dependency-free vector graphics.
void writeSvgLineChart(const std::string& path, const std::vector<double>& x, const std::vector<double>& y,
                       const std::string& title, const std::string& xLabel, const std::string& yLabel) {
    if (x.empty() || y.empty() || x.size() != y.size()) return;
    const int W = 900, H = 480, L = 80, R = 30, T = 50, B = 70;
    const double minX = *std::min_element(x.begin(), x.end());
    const double maxX = *std::max_element(x.begin(), x.end());
    const double minY = std::min(0.0, *std::min_element(y.begin(), y.end()));
    const double maxY = std::max(1.0, *std::max_element(y.begin(), y.end()));
    auto sx = [&](double v) { return L + (v - minX) / std::max(maxX - minX, 1e-12) * (W - L - R); };
    auto sy = [&](double v) { return H - B - (v - minY) / std::max(maxY - minY, 1e-12) * (H - T - B); };

    std::ofstream out(path);
    out << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H << "'>\n";
    out << "<rect width='100%' height='100%' fill='white'/>\n";
    out << "<text x='" << W / 2 << "' y='28' text-anchor='middle' font-size='20' font-family='Arial'>" << title << "</text>\n";
    out << "<line x1='" << L << "' y1='" << H - B << "' x2='" << W - R << "' y2='" << H - B << "' stroke='black'/>\n";
    out << "<line x1='" << L << "' y1='" << T << "' x2='" << L << "' y2='" << H - B << "' stroke='black'/>\n";
    out << "<polyline fill='none' stroke='black' stroke-width='2' points='";
    for (size_t i = 0; i < x.size(); ++i) out << sx(x[i]) << ',' << sy(y[i]) << ' ';
    out << "'/>\n";
    for (size_t i = 0; i < x.size(); ++i) out << "<circle cx='" << sx(x[i]) << "' cy='" << sy(y[i]) << "' r='4' fill='black'/>\n";
    out << "<text x='" << W / 2 << "' y='" << H - 20 << "' text-anchor='middle' font-size='14' font-family='Arial'>" << xLabel << "</text>\n";
    out << "<text transform='translate(20," << H / 2 << ") rotate(-90)' text-anchor='middle' font-size='14' font-family='Arial'>" << yLabel << "</text>\n";
    out << "</svg>\n";
}

void writeSvgBarChart(const std::string& path, const std::vector<double>& x, const std::vector<double>& y,
                      const std::string& title, const std::string& xLabel, const std::string& yLabel) {
    if (x.empty() || y.empty() || x.size() != y.size()) return;
    const int W = 900, H = 480, L = 80, R = 30, T = 50, B = 70;
    const double maxY = std::max(1.0, *std::max_element(y.begin(), y.end()));
    const double barW = static_cast<double>(W - L - R) / (y.size() * 1.5);
    auto sx = [&](size_t i) { return L + (i + 0.25) * (W - L - R) / y.size(); };
    auto sy = [&](double v) { return H - B - v / maxY * (H - T - B); };

    std::ofstream out(path);
    out << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H << "'>\n";
    out << "<rect width='100%' height='100%' fill='white'/>\n";
    out << "<text x='" << W / 2 << "' y='28' text-anchor='middle' font-size='20' font-family='Arial'>" << title << "</text>\n";
    out << "<line x1='" << L << "' y1='" << H - B << "' x2='" << W - R << "' y2='" << H - B << "' stroke='black'/>\n";
    out << "<line x1='" << L << "' y1='" << T << "' x2='" << L << "' y2='" << H - B << "' stroke='black'/>\n";
    for (size_t i = 0; i < y.size(); ++i) {
        out << "<rect x='" << sx(i) << "' y='" << sy(y[i]) << "' width='" << barW << "' height='" << (H - B - sy(y[i])) << "' fill='gray'/>\n";
        out << "<text x='" << sx(i) + barW / 2 << "' y='" << H - B + 18 << "' text-anchor='middle' font-size='11' font-family='Arial'>" << x[i] << "</text>\n";
    }
    out << "<text x='" << W / 2 << "' y='" << H - 20 << "' text-anchor='middle' font-size='14' font-family='Arial'>" << xLabel << "</text>\n";
    out << "<text transform='translate(20," << H / 2 << ") rotate(-90)' text-anchor='middle' font-size='14' font-family='Arial'>" << yLabel << "</text>\n";
    out << "</svg>\n";
}

void writePricingComparisonSvg(const std::string& path, const std::vector<SummaryRow>& rows) {
    if (rows.empty()) return;
    const int W = 1000, H = 520, L = 80, R = 30, T = 50, B = 110;
    double maxY = 1.0;
    for (const auto& r : rows) maxY = std::max({maxY, r.fairValue, r.binomial, r.monteCarlo});
    const double groupW = static_cast<double>(W - L - R) / rows.size();
    const double barW = groupW / 5.0;
    auto sy = [&](double v) { return H - B - v / maxY * (H - T - B); };

    std::ofstream out(path);
    out << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H << "'>\n";
    out << "<rect width='100%' height='100%' fill='white'/>\n";
    out << "<text x='" << W / 2 << "' y='28' text-anchor='middle' font-size='20' font-family='Arial'>Japan Derivatives Pricing Comparison</text>\n";
    out << "<line x1='" << L << "' y1='" << H - B << "' x2='" << W - R << "' y2='" << H - B << "' stroke='black'/>\n";
    out << "<line x1='" << L << "' y1='" << T << "' x2='" << L << "' y2='" << H - B << "' stroke='black'/>\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const double base = L + i * groupW + groupW * 0.2;
        const std::array<double, 3> vals = {rows[i].fairValue, rows[i].binomial, rows[i].monteCarlo};
        const std::array<std::string, 3> fills = {"#444", "#888", "#bbb"};
        for (int j = 0; j < 3; ++j) {
            const double x = base + j * barW;
            out << "<rect x='" << x << "' y='" << sy(vals[j]) << "' width='" << barW * 0.9 << "' height='" << (H - B - sy(vals[j])) << "' fill='" << fills[j] << "'/>\n";
        }
        out << "<text transform='translate(" << base + barW << ',' << H - B + 18 << ") rotate(25)' font-size='11' font-family='Arial'>" << rows[i].instrument << "</text>\n";
    }
    out << "<text x='" << W - 230 << "' y='60' font-size='12' font-family='Arial'>Fair Value / Binomial / Monte Carlo</text>\n";
    out << "</svg>\n";
}

// ---------------------------------------------------------------------
// Main framework orchestration
// ---------------------------------------------------------------------

void runFramework(const std::string& configPath) {
    Config cfg = loadConfig(configPath);
    ensureDirs({cfg.dataDir, cfg.outputDir, cfg.plotDir});

    const auto optionChain = loadOptionChain(cfg.optionChainCsv);
    std::vector<SummaryRow> summaryRows;
    std::vector<std::pair<std::string, Signal>> signalRows;
    std::map<std::string, std::vector<ScenarioRow>> scenarioTables;
    const Date tradeDate = todayLocal();

    for (const auto& instrument : cfg.instruments) {
        std::cout << "\nProcessing: " << instrument.name << "\n";

        std::vector<PricePoint> prices;
        const bool useYahoo = upper(cfg.quoteSource) == "YFINANCE";
        if (useYahoo) {
            auto liveHistory = fetchYahooHistory(instrument.yahoo, cfg.startDate);
            if (liveHistory) prices = *liveHistory;
        }

        const std::string historyPath = cfg.dataDir + "/" + safeName(instrument.name) + "_historical_prices.csv";
        if (prices.empty()) {
            try {
                prices = loadHistoricalCsv(historyPath);
                std::cout << "Loaded local historical data: " << historyPath << "\n";
            } catch (const std::exception& e) {
                std::cout << "Skipping " << instrument.name << ": historical data unavailable. "
                          << "Provide " << historyPath << " or enable curl/Yahoo access.\n";
                continue;
            }
        } else {
            std::ofstream out(historyPath);
            out << "date,close\n";
            for (const auto& p : prices) out << p.date << ',' << formatDouble(p.close, 8) << '\n';
        }

        ReturnStats stats;
        try {
            stats = calculateReturnStats(prices, cfg.tradingDays);
        } catch (const std::exception& e) {
            std::cout << "Skipping " << instrument.name << ": " << e.what() << "\n";
            continue;
        }

        double spotOrFuture = prices.back().close;
        std::string quoteTimestamp = prices.back().date;
        if (useYahoo) {
            auto quote = fetchLatestYahooQuote(instrument.yahoo);
            if (quote) {
                spotOrFuture = quote->first;
                quoteTimestamp = quote->second;
            }
        }

        auto option = selectAtmOption(optionChain, instrument, spotOrFuture, tradeDate);
        if (!option) {
            std::cout << "No option data found for " << instrument.name << ". Skipping.\n";
            continue;
        }

        const double timeToExpiry = std::max(daysBetween(tradeDate, option->expiry) / 365.0, 1.0 / 365.0);
        const double iv = impliedVolatility(option->marketPrice, spotOrFuture, option->strike, timeToExpiry,
                                            cfg.riskFreeRate, option->optionType, instrument.assetType,
                                            instrument.dividendYield);
        const double sigma = isFinite(iv) && iv > 0.0 ? iv : stats.historicalVol;

        const double fairValue = modelPrice(spotOrFuture, option->strike, timeToExpiry, cfg.riskFreeRate,
                                            sigma, option->optionType, instrument.assetType,
                                            instrument.dividendYield);
        const double binomial = binomialTreePrice(spotOrFuture, option->strike, timeToExpiry, cfg.riskFreeRate,
                                                  sigma, 500, option->optionType, instrument.assetType,
                                                  instrument.dividendYield);
        const McResult mc = monteCarloPrice(spotOrFuture, option->strike, timeToExpiry, cfg.riskFreeRate,
                                            sigma, option->optionType, instrument.assetType,
                                            instrument.dividendYield, cfg.mcPaths, cfg.mcSteps, cfg.mcSeed);
        const Greeks greeks = finiteDifferenceGreeks(spotOrFuture, option->strike, timeToExpiry, cfg.riskFreeRate,
                                                     sigma, option->optionType, instrument.assetType,
                                                     instrument.dividendYield);
        const Signal signal = generateSignal(option->marketPrice, fairValue, greeks.delta, cfg.mispricingThreshold);
        const auto scenarios = scenarioAnalysis(spotOrFuture, option->strike, timeToExpiry, cfg.riskFreeRate,
                                                sigma, option->optionType, instrument.assetType,
                                                instrument.dividendYield);

        scenarioTables[instrument.name] = scenarios;
        writeScenarioCsv(cfg.dataDir + "/" + safeName(instrument.name) + "_scenario_analysis.csv", scenarios);

        SummaryRow row;
        row.instrument = instrument.name;
        row.symbol = instrument.symbol;
        row.assetType = instrument.assetType;
        row.yahoo = instrument.yahoo;
        row.tradingview = instrument.tradingview;
        row.quoteTimestamp = quoteTimestamp;
        row.spotOrFuture = roundN(spotOrFuture, 4);
        row.expiry = option->expiry.iso();
        row.strike = roundN(option->strike, 4);
        row.optionType = option->optionType;
        row.marketPrice = roundN(option->marketPrice, 4);
        row.historicalVol = roundN(stats.historicalVol, 6);
        row.impliedVol = isFinite(iv) ? roundN(iv, 6) : NaN;
        row.volUsed = roundN(sigma, 6);
        row.fairValue = roundN(fairValue, 4);
        row.binomial = roundN(binomial, 4);
        row.monteCarlo = roundN(mc.price, 4);
        row.mcError = roundN(mc.error, 6);
        row.delta = roundN(greeks.delta, 6);
        row.gamma = roundN(greeks.gamma, 8);
        row.vega = roundN(greeks.vega, 6);
        row.theta = roundN(greeks.theta, 6);
        row.rho = roundN(greeks.rho, 6);
        row.annualizedReturn = roundN(stats.annualizedReturn, 6);
        row.skewness = roundN(stats.skewness, 6);
        row.kurtosis = roundN(stats.kurtosis, 6);
        summaryRows.push_back(row);
        signalRows.push_back({instrument.name, signal});
    }

    writePricingSummaryCsv(cfg.outputDir + "/pricing_summary.csv", summaryRows);
    writeTradingSignalsCsv(cfg.outputDir + "/trading_signals.csv", signalRows);
    writePricingComparisonSvg(cfg.plotDir + "/pricing_comparison.svg", summaryRows);

    auto [thresholdSummary, allThresholdSignals] = runThresholdOptimization(summaryRows);
    writeThresholdSummaryCsv(cfg.outputDir + "/threshold_optimization_summary.csv", thresholdSummary);
    writeAllThresholdSignalsCsv(cfg.outputDir + "/all_threshold_signals_1_to_10.csv", allThresholdSignals);

    if (!thresholdSummary.empty()) {
        const auto bestIt = std::max_element(thresholdSummary.begin(), thresholdSummary.end(),
            [](const ThresholdSummaryRow& a, const ThresholdSummaryRow& b) {
                return a.optimizationScore < b.optimizationScore;
            });
        const double bestThreshold = bestIt->threshold;
        std::vector<ThresholdSignalRow> bestRows;
        for (const auto& r : allThresholdSignals) {
            if (std::abs(r.threshold - bestThreshold) < 1e-12) bestRows.push_back(r);
        }
        writeAllThresholdSignalsCsv(cfg.outputDir + "/trading_signals_best_threshold.csv", bestRows);

        std::vector<double> x, score, trades;
        for (const auto& r : thresholdSummary) {
            x.push_back(r.thresholdPercent);
            score.push_back(r.optimizationScore);
            trades.push_back(static_cast<double>(r.tradeSignals));
        }
        writeSvgLineChart(cfg.plotDir + "/threshold_optimization_score.svg", x, score,
                          "Threshold Optimization Score: 1% to 10%", "Mispricing Threshold (%)", "Optimization Score");
        writeSvgBarChart(cfg.plotDir + "/trade_signals_by_threshold.svg", x, trades,
                         "Number of Trade Signals at Different Thresholds", "Mispricing Threshold (%)", "Number of Trade Signals");

        std::cout << "\nBest threshold: " << formatDouble(bestThreshold * 100.0, 2) << "%"
                  << " | Trade signals: " << bestIt->tradeSignals
                  << " | Optimization score: " << formatDouble(bestIt->optimizationScore, 4) << "\n";
    }

    generateReport(cfg.outputDir + "/research_report.md", summaryRows, signalRows, scenarioTables, thresholdSummary);

    std::cout << "\nProject completed.\n";
    std::cout << "Pricing summary saved to: " << cfg.outputDir << "/pricing_summary.csv\n";
    std::cout << "Trading signals saved to: " << cfg.outputDir << "/trading_signals.csv\n";
    std::cout << "Threshold optimization saved to: " << cfg.outputDir << "/threshold_optimization_summary.csv\n";
    std::cout << "Report saved to: " << cfg.outputDir << "/research_report.md\n";

    std::cout << "\nTrading Signals:\n";
    if (signalRows.empty()) {
        std::cout << "No signals generated.\n";
    } else {
        for (const auto& [inst, sig] : signalRows) {
            std::cout << std::left << std::setw(28) << inst
                      << std::setw(28) << sig.signal
                      << std::setw(8) << sig.action
                      << "edge=" << formatDouble(sig.edgePct, 6) << '\n';
        }
    }
}

} // namespace quant

int main(int argc, char** argv) {
    std::string configPath = "config/config.yaml";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: japan_derivatives [--config config/config.yaml]\n";
            return 0;
        }
    }

    try {
        quant::runFramework(configPath);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
