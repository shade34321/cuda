#include <stdio.h>

// Prints info about the device
// Takes in the device number and a pointer to the properties
void printProperties(int i, cudaDeviceProp *prop){
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop->name );
        printf( "Compute capability:  %d.%d\n", prop->major, prop->minor );
        printf( "Clock rate:  %d\n", prop->clockRate );
        printf( "Device copy overlap:  " );
        if (prop->deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n");
        printf( "Kernel execution timeout :  " );
        if (prop->kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );

        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop->totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop->totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop->memPitch );
        printf( "Texture Alignment:  %ld\n", prop->textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    prop->multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop->sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop->regsPerBlock );
        printf( "Threads in warp:  %d\n", prop->warpSize );
        printf( "Max threads per block:  %d\n",
                    prop->maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop->maxThreadsDim[0], prop->maxThreadsDim[1],
                    prop->maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop->maxGridSize[0], prop->maxGridSize[1],
                    prop->maxGridSize[2] );
        printf( "\n" );
}

int main( void ) {
	cudaDeviceProp  prop;
	int count;

	//Doesn't handle errors
	cudaGetDeviceCount(&count);

	printf("Device Count: %d\n", count);

	//while(count > 0){
	for(int i = 0; i < count; i++){
		//Doesn't handle errors
		printf("Device Count: %d\n", i);
		cudaGetDeviceProperties( &prop, i );
		printProperties(i, &prop);	
		//count--;		
	}

	return 0;
}
