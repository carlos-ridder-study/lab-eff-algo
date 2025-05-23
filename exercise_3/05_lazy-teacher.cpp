#include <vector>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <cmath>
#include <iostream>
#include <chrono>
#include <bitset>
#include <iomanip>

// Efficiently computes the probability that exactly k out of 2k people vote yes
// Uses dynamic programming, matching the Python implementation
double compute_tie_probability(const std::vector<double>& probabilities, const std::vector<int>& subset) {
    int k = subset.size();
    std::vector<double> dp(k + 1, 0.0);
    dp[0] = 1.0;
    for (int idx : subset) {
        double p = probabilities[idx];
        for (int s = k; s >= 1; --s) {
            dp[s] = dp[s] * (1.0 - p) + dp[s - 1] * p;
        }
        dp[0] *= (1.0 - p);
        // Early exit if all probabilities are zero
        bool all_zero = true;
        for (int s = 0; s <= k; ++s) {
            if (dp[s] != 0.0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) return 0.0;
    }
    return dp[k / 2];
}

std::pair<double, std::vector<int>> local_search(
    const std::vector<int>& selected,
    double tie_prob,
    const std::vector<double>& probs,
    int k,
    double temperature = 0.0
) {
    int n = probs.size();
    std::vector<int> best_subset = selected;
    double best_prob = tie_prob;
    std::vector<int> candidates;
    candidates.reserve(n - k);
    std::unordered_set<int> selected_set(selected.begin(), selected.end());
    for (int i = 0; i < n; ++i) {
        if (selected_set.find(i) == selected_set.end()) {
            candidates.push_back(i);
        }
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> shuffled_selected = selected;
    std::shuffle(shuffled_selected.begin(), shuffled_selected.end(), g);
    std::shuffle(candidates.begin(), candidates.end(), g);
    for (size_t i = 0; i < shuffled_selected.size(); ++i) {
        for (int j : candidates) {
            if (std::find(shuffled_selected.begin(), shuffled_selected.end(), j) != shuffled_selected.end()) continue;
            int old = shuffled_selected[i];
            shuffled_selected[i] = j;
            double new_prob = compute_tie_probability(probs, shuffled_selected);
            double delta = new_prob - best_prob;
            if (delta > 0) {
                best_prob = new_prob;
                best_subset = shuffled_selected;
                shuffled_selected[i] = old; // revert for next iteration
                return local_search(best_subset, best_prob, probs, k, 0.9 * temperature);
            } else if (delta < 0 && temperature > 0) {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double p = dist(g);
                if (p < std::exp(delta / temperature)) {
                    best_prob = new_prob;
                    best_subset = shuffled_selected;
                    shuffled_selected[i] = old; // revert for next iteration
                    return local_search(best_subset, best_prob, probs, k, 0.9 * temperature);
                }
            }
            shuffled_selected[i] = old; // revert
        }
    }
    return {best_prob, best_subset};
}

// Brute-force: Try all subsets of size 2k and return the best tie probability and subset
std::pair<double, std::vector<int>> brute_force_best_tie(const std::vector<double>& probs, int k) {
    int n = probs.size();
    int subset_size = k;
    double best_prob = -1.0;
    std::vector<int> best_subset;
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;
    std::vector<bool> select(n, false);
    for (int i = 0; i < subset_size; ++i) select[i] = true;
    do {
        std::vector<int> subset;
        for (int i = 0; i < n; ++i) {
            if (select[i]) subset.push_back(i);
        }
        double prob = compute_tie_probability(probs, subset);
        if (prob > best_prob) {
            best_prob = prob;
            best_subset = subset;
        }
    } while (std::prev_permutation(select.begin(), select.end()));
    return {best_prob, best_subset};
}

std::string format_double(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << value;
    std::string s = oss.str();
    // Remove trailing zeros and possibly the decimal point
    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    if (!s.empty() && s.back() == '.') s.pop_back();
    return s;
}

int main() {
    int n, k;
    int N = 30; // Number of repetitions
    //std::cout << "Enter n and k: ";
    std::cin >> n >> k;
    std::vector<double> probs;
    probs.reserve(n);
    if(n > 0){
        
        //std::cout << "Enter " << n << " probabilities (space-separated): ";
        for (int i = 0; i < n; ++i) {
            double p;
            std::cin >> p;
            probs.push_back(p);
        }   

        // Initial selection: first k indices
        std::vector<int> selected(k);
        for (int i = 0; i < k; ++i) selected[i] = i;
        double tie_prob = compute_tie_probability(probs, selected);
        tie_prob = local_search(selected, tie_prob, probs, k).first;
        std::cout << format_double(tie_prob) << "\n";
        //std::cout << "Optimal tie probability: " << format_double(prob_opt) << "\n";
    }
    else{
    // For benchmarking: generate random probabilities if n==0
        n = 23;
        k = 13;
        probs.resize(n);
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < n; ++i) probs[i] = dist(g);
        std::cout << "Generated " << n << " random probabilities for benchmarking.\n";
        int match_count = 0;
        double total_local_time = 0.0;
        double total_brute_time = 0.0;
        double total_local_prob = 0.0;
        double total_brute_prob = 0.0;
        
        for (int rep = 0; rep < N; ++rep) {
            std::vector<int> all_indices(n);
            for (int i = 0; i < n; ++i) all_indices[i] = i;
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(all_indices.begin(), all_indices.end(), g);
            std::vector<int> selected(all_indices.begin(), all_indices.begin() + k);
            double tie_prob = compute_tie_probability(probs, selected);
            std::unordered_set<int> selected_set(selected.begin(), selected.end());
            std::vector<int> candidates;
            //std::random_device rd;
            //std::mt19937 g(rd());
            auto start_local = std::chrono::high_resolution_clock::now();
            auto local_result = local_search(selected, tie_prob, probs, k, 0.0);
            auto end_local = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_local = end_local - start_local;
            total_local_time += elapsed_local.count();
            total_local_prob += local_result.first;

            auto start_brute = std::chrono::high_resolution_clock::now();
            auto brute_result = brute_force_best_tie(probs, k);
            auto end_brute = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_brute = end_brute - start_brute;
            total_brute_time += elapsed_brute.count();
            total_brute_prob += brute_result.first;

            if (std::abs(local_result.first - brute_result.first) < 1e-9) {
                match_count++;
            }
            }
        

        std::cout << "Average local search time: " << (total_local_time / N) << " seconds\n";
        std::cout << "Average brute-force time: " << (total_brute_time / N) << " seconds\n";
        std::cout << "Local search matches brute-force: " << (100.0 * match_count / N) << "% of the time\n";
        std::cout << "Average local search probability: " << (total_local_prob / N) << "\n";
        std::cout << "Average brute-force probability: " << (total_brute_prob / N) << "\n";
        }
    return 0;
}