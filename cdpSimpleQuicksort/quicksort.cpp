#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

typedef unsigned long long timestamp_t;

void initialize_data(unsigned int *dst, unsigned int nitems) {
    std::srand(4294967295 );

    for(unsigned i = 0; i < nitems; i++) {
        dst[i] = std::rand() % nitems;
    }
}

void check_results(int n, unsigned int * results) {
    for(int i = 1; i < n; ++i) {
        if(results[i-1] > results[i]) {
            std::cout << "Invalid item[" << i-i << "]: " << results[i-1] << " greater than " << results[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "OK" << std::endl;
}

void log_results(int num, timestamp_t timings[], int length) {
    char filename[31];
    snprintf(filename, 30, "serial_sorting_time_results.csv");

    std::ofstream data_log;
    data_log.open(filename, std::ios::out | std::ios::app);

    data_log << num;

    for(int i = 0; i < length; i++) {
        data_log << "," << timings[i];
    }

    data_log << std::endl;
    data_log.close();
}

static timestamp_t get_timestamp() {
    struct timeval now;
    gettimeofday(&now, NULL);
    return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

int main(int argc, char **argv) {

    return 0;
}
