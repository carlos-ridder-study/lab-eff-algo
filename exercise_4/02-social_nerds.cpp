#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace std;


int main(){
    int n;
    string line;
    getline(cin, line);
    n = stoi(line);
    vector<int> low(n);
    vector<int> high(n);
    vector<bool> dp(1LL << n, false);
    dp[0] = true;
    vector<vector<int>> groups(n + 1);
    
    for (int i = 0; i < n; ++i) {
        // Read the low and high values for each nerd
        getline(cin, line);
        istringstream iss(line);
        iss >> low[i] >> high[i];
        for (int k = low[i]; k <= high[i]; ++k) {
            groups[k].push_back(i);
        }
    }

    int min_groupsize = n;
    for(int k=0; k<=n; ++k){
        // for all possible groupsizes, iterate through all possible
        // groups of that size and set dp[group] = true
        if (groups[k].empty() or (int)groups[k].size() < k) continue;
        min_groupsize = min(min_groupsize, k);
        vector<bool> select(groups[k].size(), false);
        fill(select.end() - k, select.end(), true);
        do {
            int subset = 0;
            for (int j = 0; j < (int)groups[k].size(); ++j) {
                if (select[j]) subset |= (1LL << groups[k][j]);
            }
            dp[subset] = true;
        } while (next_permutation(select.begin(), select.end()));
    }

for (int subset = 0; subset < (1LL << n); ++subset) {
    int size = __builtin_popcount(subset);
    if (size < min_groupsize) continue;

    // Iterate through all non-empty proper subsets s' of subset
    // To only consider half, process s' only if s' <= (subset ^ s')
    for (int s = subset; s; s = (s - 1) & subset) {
        int s_size = __builtin_popcount(s);
        if (s_size < min_groupsize) continue;
        int t = subset ^ s;
        if (s > t) continue; // only process half (avoid double-counting)
        if (dp[s] && dp[t]) {
            dp[subset] = true;
            break; // no need to check further subsets
        }
    }
}

// Output the result
cout << (dp[(1LL << n) - 1] ? "possible" : "impossible") << endl;

/**
auto end = chrono::high_resolution_clock::now();
chrono::duration<double, milli> elapsed = end - start;
// Output the time taken for the first method
cout << "Time taken for first method: " << elapsed.count() << " ms, count: " << count << ", result " <<  (dp[(1LL << n) - 1] ? "possible" : "impossible") << endl;


dp = dp_orig; // Reset dp to original state
auto start1 = chrono::high_resolution_clock::now();
int count1 = 0;
for (int size = min_groupsize; size <= n; ++size) {
    cout << "Processing groups of size " << size << endl;
    vector<bool> select(n, false);
    fill(select.end() - size, select.end(), true);
    do {
        int subset = 0;
        vector<int> subset_indices;
        for (int j = 0; j < n; ++j) {
            if (select[j]) {
                subset |= (1LL << j);
                subset_indices.push_back(j);
            }
        }
        
        for(int s_size = min_groupsize; s_size <= size/2; ++s_size) {
            vector<bool> select_s(size, false);
            fill(select_s.end() - s_size, select_s.end(), true);
            do {
                ++count1;
                int s = 0;
                for (int j = 0; j < n; ++j) {
                    if (select_s[j]) s |= (1LL << subset_indices[j]);
                }
                int t = subset ^ s;
                if(dp[s] && dp[t]) {
                    dp[subset] = true;
                    goto NEXT_SUBSET; // no need to check further subsets
                }
            } while (next_permutation(select_s.begin(), select_s.end()));
        }
        NEXT_SUBSET: ;
    } while (next_permutation(select.begin(), select.end()));
}
auto end1 = chrono::high_resolution_clock::now();
chrono::duration<double, milli> elapsed1 = end1 - start1;
// Output the time taken for the second method
cout << "Time taken for second method: " << elapsed1.count() << " ms, count: " << count1 << ", result " <<  (dp[(1 << n) - 1] ? "possible" : "impossible") << endl;
*/

/**
    for(int k=0; k<=n; ++k){
        if(!groupsize_bitsets[k].empty()){
            cout << "groupsize " << k << ": ";
            for(int i=0; i<n; ++i){
                if(groupsize_bitsets[k][i]){
                    cout << "1";
                }
                else{
                    cout << "0";
                }
            }
            endl(cout);
        }
    }
*/
    return 0;
}