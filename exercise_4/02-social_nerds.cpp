#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace std;

bool partition(unordered_set<int>& S, int k, vector<int>& low, vector<int>& high, vector<vector<int>>& groupsizes){
    // Base case: if k is 0, we have successfully partitioned the set
    if (S.empty()) {
        return true;
    }
    if (k == 0) {
        return false;
    }
    vector<int> G;
    for(int i : groupsizes[k]){
        if(S.find(i) != S.end()){
            G.push_back(i);
        }
    }
    bool partition_possible = false;
    int l = G.size();
    if(l >= k){
        unordered_set<int> S_ = S;
        for (int i = 0; i < k; ++i) {
            S_.erase(G[i]);
        }
        partition_possible = partition(S_, k, low, high, groupsizes);
    }
    if(partition_possible){
        return true;
    }
    else{
        return partition(S, k - 1, low, high, groupsizes);
    }
}

int main(){
    int n;
    string line;
    getline(cin, line);
    n = stoi(line);
    vector<int> low(n);
    vector<int> high(n);
    unordered_set<int> S;
    vector<vector<int>> groupsizes(n + 1);
    
    for (int i = 0; i < n; ++i) {
        // Read the low and high values for each nerd
        getline(cin, line);
        istringstream iss(line);
        iss >> low[i] >> high[i];
        for (int k = low[i]; k <= high[i]; ++k) {
            groupsizes[k].push_back(i);
        }
        S.insert(i);
    }
    for(int k = 0; k <= n; k++) {
        // sort in ascending order of lower bound
        sort(groupsizes[k].begin(), groupsizes[k].end(), [&low](int a, int b) {
        return low[a] > low[b];
    });
    }

    bool partition_possible = partition(S, n, low, high, groupsizes);
    cout << (partition_possible ? "possible" : "impossible") << endl;
    return 0;
}