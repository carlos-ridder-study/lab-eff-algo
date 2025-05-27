#include <iostream>

int main()
{
    int minOpen = 0; //minimum number of brackets open
    int maxOpen = 0; //maximum number of brackets open
    char symbol;
    
    while (std::cin.get(symbol)){
        //std::cout << symbol;
        switch(symbol){
            case '(':
                minOpen++;
                maxOpen++;
                break;
            case '?': 
                minOpen--;
                maxOpen++;
                if (minOpen < 0) minOpen = 0;
                break;
            case ')': 
                minOpen--;
                maxOpen--;
                if (minOpen < 0) minOpen = 0; //cannot be smaller than 0
                if (maxOpen < 0) { //unmatched numer of brackets
                    std::cout << '0' << std::endl;
                    return 0;
                }
                break;
                
            case '\n':
                break;
            case '\r':
                break;
        }
    }
    if (minOpen == 0 && maxOpen % 2 == 0) {
        std::cout << '1' << std::endl;
    } else {
        std::cout << '0' << std::endl;
    }
    
    return 0;
}
