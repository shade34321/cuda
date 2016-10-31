#include <iostream>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include <sys/time.h>

#define MAX_DEPTH       16
#define INSERTION_SORT  32

typedef unsigned long long timestamp_t;

/////////////////////////////////////////////////////////////////////////////
// I'm following the way the example from Nvidia's quick sort works.
// When it's get to deep or when we drop below a certain point we will
// perform selection sort.
/////////////////////////////////////////////////////////////////////////////

void selection_sort(unsigned int *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        unsigned int min_val = data[i];
        int min_idx = i;

        for (int j = i+1; j <= right; ++j) {
            unsigned int val_j = data[j];

            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }

        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// To avoid any speedups which Nvidia didn't use which might invalidate the
// results I'm going to follow their algorithm fairly closely. 
//
/////////////////////////////////////////////////////////////////////////////

void quicksort(unsigned int *data, int left, int right, int depth) {
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

    unsigned int *lptr = data+left;
    unsigned int *rptr = data+right;
    unsigned int pivot = data[(left+right)/2];

    while(lptr <= rptr) {
        unsigned int lval = *lptr;
        unsigned int rval = *rptr;

        while (lval < pivot) {
            lptr++;
            lval = *lptr;
        }

        while (rval > pivot) {
            rptr--;
            rval = *rptr;
        }

        if (lptr <= rptr) {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    int nright = rptr - data;
    int nleft = lptr - data;

    if (left < (rptr-data)) {
        quicksort(data, left, nright, depth+1);
    }

    if ((lptr-data) < right) {
        quicksort(data, nleft, right, depth+1);
    }
}

void run_qsort(unsigned int *data, unsigned int nitems) {
    int left = 0;
    int right = nitems - 1;
    quicksort(data, left, right, 0);
}

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
    char filename[35];
    snprintf(filename, 35, "serial_sorting_time_results.csv");

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

void printData(unsigned int *data, unsigned int nitems){
    for (unsigned int i = 0; i < nitems; i++) {
        std::cout << data[i] << " ";
        if (i % 80 == 0 && i != 0) {
            std::cout << std::endl; //So we only print 80 characters per line.
        }
    }

    std::cout << std::endl;
}

int main(int argc, char **argv) {
    int num_items = 128;
    timestamp_t timings[4];
    unsigned int *data = 0;

    if (argc == 2) {
        std::stringstream convert(argv[1]);
        if (!(convert >> num_items)) {
            std::cout << "Please enter a number..." << std::endl;
            return 1;
        }
        std::cout << "We will be using " << num_items << std::endl;
    }

    for (int i = 0; i < 4; i++) {
        data = (unsigned int *)malloc(num_items*sizeof(unsigned int));
        initialize_data(data, num_items);

        //printData(data, num_items);

        std::cout << "Running quicksort on " << num_items << " elements." << std::endl;
        timestamp_t t0 = get_timestamp();
        run_qsort(data, num_items);
        
        timestamp_t t1 = get_timestamp();

        //printData(data, num_items);

        timings[i] = t1 - t0;
        timings[3] += timings[i];

        std::cout << "Validating results: ";
        check_results(num_items, data);

        free(data);
    }
    timings[3] /= 3;

    log_results(num_items, timings, 4);
    return 0;
}
