#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>


using namespace std;

int FRAME_DELAY_MS = 800;  // delay between frames
constexpr const char* CLR_RESET = "\033[0m";
constexpr const char* CLR_RED   = "\033[31m";
constexpr const char* CLR_GREEN = "\033[32m";
constexpr const char* CLR_BLUE  = "\033[34m";

// Print one frame: clears screen, then draws the grid
void print_grid(uint64_t path_mask, vector<int>& stops, int rows, int cols) {
    // clear screen & move cursor to home
    std::cout << "\033[2J\033[H";

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            bool on_path = (path_mask >> idx) & 1ULL;

            // decide what to print
            for(int i = 0; i < stops.size(); ++i) {
                if (idx == stops[i]) {
                    if (on_path) {
                        cout << ' ' << CLR_GREEN << i << CLR_RESET;
                    }
                    else {
                        cout << ' ' << CLR_BLUE << i << CLR_RESET;
                    }
                    goto next_cell;
                }
            }
            if (on_path) {
                std::cout << CLR_GREEN << " *" << CLR_RESET;
            }
            else {
                std::cout << " .";
            }
            next_cell:;
        }
        std::cout << "\n";
    }
    // pause so you can see it
    std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_DELAY_MS));
}

/**
 * Computes the Manhattan distance between two grid cells given by their bit
 * representations b = i * cols + j for cell (i,j).
 */
int distance(int bit1, int bit2, int cols) {
    int i1, j1, i2, j2;
    j1 = bit1 % cols;
    j2 = bit2 % cols;
    i1 = (bit1-j1) / cols;
    i2 = (bit2-j2) / cols;
    return abs(i1 - i2) + abs(j1 - j2);
}

static std::vector<int> get_neighbors(int v, int rows, int cols) {
    int i, j;
    j = v % cols;
    i = (v - j) / cols;
    std::vector<int> neighbors;
    // up
    if (i > 0)        neighbors.push_back((i - 1) * cols + j);
    // down
    if (i + 1 < rows) neighbors.push_back((i + 1) * cols + j);
    // left
    if (j > 0)        neighbors.push_back(i * cols + (j - 1));
    // right
    if (j + 1 < cols) neighbors.push_back(i * cols + (j + 1));
    return neighbors;
}

void dfs(int cur, uint64_t visited, int steps_left, int segment, vector<uint64_t>& paths,
            vector<int>& stops, vector<int>& path_lengths, int rows, int cols) {
    
    // visualization
    // print_grid(visited, stops, rows, cols);

    int target = stops[segment+1];
    int d = distance(cur,target, cols);
    if ((d > steps_left) || ((d % 2) != (steps_left % 2))) return;
    if (steps_left==0) {
        if (cur==target) {
            if (segment < stops.size() - 2) {
                // Move to the next segment
                segment ++;
                int cur = stops[segment];
                int steps_left = path_lengths[segment];
                dfs(cur, visited, steps_left, segment, paths, stops, path_lengths, rows, cols);
            }
            else {
                // Reached the last stop, add the path to the result
                paths.push_back(visited);
            }
        } 
        return;
    }
    // explore neighbors
    vector<int> neighbors = get_neighbors(cur, rows, cols);
    for (int n : neighbors) {
        uint64_t bit = (1ULL << n);
        uint64_t new_visited = visited | bit;
        // skip if already visited
        if (visited & bit) continue;
        // skip if stop reached too early
        if (n == target && steps_left > 1) continue;
        for (int i = segment+2; i < stops.size(); ++i) {
            if (n == stops[i]) goto next_neighbor;  // can't touch future stops
        }

        // check if any other neighbors would be cut off and prune in that case
        for (int u : neighbors){
            if(u == n) continue;
            if(visited & (1ULL << u)) continue;
            // check if neighbor u would be cut off
            int unvisited = 0;
            for(int w: get_neighbors(u, rows, cols)){
                if( !(new_visited & (1ULL << w)) ) unvisited++;
            }
            if(unvisited < 2){
                if(u != stops[stops.size()-1])  goto next_neighbor;
                else if(unvisited < 1)  goto next_neighbor;
            }

        }
        dfs(n, visited | bit, steps_left - 1, segment, paths, stops, path_lengths, rows, cols);
        next_neighbor:;
    }
}


int main(){
    string line;
    getline(cin, line);
    istringstream iss(line);
    int r, c, total;
    iss >> r >> c;
    total = r * c;
    getline(cin, line);
    istringstream iss2(line);
    int r1, c1, r2, c2, r3, c3;
    iss2 >> r1 >> c1 >> r2 >> c2 >> r3 >> c3;

    // Convert stop coordinates to bit representations
    int t0, t1, t2, t3, t4;
    t0 = 0;
    t1 = r1 * c + c1;
    t2 = r2 * c + c2;
    t3 = r3 * c + c3;
    t4 = 1;
    vector<int> stops = {t0, t1, t2, t3, t4};

    // Calculate path lenghts between stops
    int l1, l2, l3, l4;
    l1 = total / 4;
    l2 = (total / 2) - l1;
    l3 = ((3 * total) / 4) - (l1 + l2);
    l4 = total - (l1 + l2 + l3);
    vector<int> path_lengths = {l1, l2, l3, l4};

    uint64_t visited = 1ULL << t0; // Start with the first stop visited
    vector<uint64_t> paths = {};
    dfs(0, visited, path_lengths[0]-1, 0, paths, stops, path_lengths, r, c);
    // Output the number of unique paths
    cout << paths.size() << endl;
    return 0;
}