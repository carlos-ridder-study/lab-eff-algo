#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stdexcept> // For exception handling

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

std::vector<boost::dynamic_bitset<>> read_input() {
    std::string line;
    int r;

    // 1. Read the first line for dimensions
    std::getline(std::cin, line);
    line = trim(line);
    std::istringstream iss(line);
    iss >> r;

    std::vector<boost::dynamic_bitset<>> rows(r, boost::dynamic_bitset<>(r, 0));

    // 2. Read in the r rows
    for(int i = 0; i < r; ++i) {
        std::getline(std::cin, line);
        line = trim(line);
        std::istringstream iss(line);
        for(int j = 0; j < r; ++j) {
            char c = line[j];
            rows[i][j] = (c == '1');
        }
    }
    return rows;
}

int main(){
    std::vector<boost::dynamic_bitset<>> rows = read_input();
    long long count = 0;
    long long r = rows.size();
    for(int i = 0; i < r; ++i) {
        for(int j = i+1; j < r; ++j) {
            // std::cout << "row " << i << ": " << rows[i] << std::endl << "row " << j << ": " << rows[j] << std::endl;
            // perform bitwise AND to count common corners marked with 1
            boost::dynamic_bitset<> result = rows[i] & rows[j];
            // std::cout << "result: " << result << std::endl;
            size_t num_corners = result.count();
            // add number of different rectangles formed by these two rows
            // (n choose 2) = n * (n - 1) / 2
            count += (long long) (num_corners * (num_corners - 1)) / 2;
        }
    }
    std::cout << count << std::endl;
    return 0;
}