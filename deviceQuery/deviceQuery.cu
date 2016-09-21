#include <stdio.h>

// Prints info about the device
// Takes in the device number and a pointer to the properties
void printProperties(int i, cudaDeviceProp *prop){
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop->name );
}

int main( void ) {
	cudaDeviceProp  *prop;
	int count;

	//Doesn't handle errors
	cudaGetDeviceCount(&count);

	printf("Device Count: %d\n", count);

	//while(count > 0){
	for(int i = 0; i < count; i++){
		//Doesn't handle errors
		cudaGetDeviceProperties( prop, i );
		printProperties(i, prop);	
		//count--;		
	}

	return 0;
}
