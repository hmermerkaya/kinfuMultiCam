#include "device.hpp"


namespace kinfu
{
	namespace device
	{

			__global__ void
			loadTsdfValueKernel (short2* first_point_pointer, float4 *data_ptr, int number_of_points, const int divisor, kinfu::gpu::tsdf_buffer buffer)
			{
				int i = blockDim.x * blockIdx.x  + threadIdx.x;
				if (i < number_of_points)
				{
					float4 *point_ptr = data_ptr + (i * 2);
					float4 *tsdfvalue_ptr = point_ptr + 1;

					if( //Indices x,y,z are within [0...VOLUME_{X,Y,Z}]
							( ( (int) ( (*point_ptr).x ) < buffer.voxels_size.x) && ( (int) ( (*point_ptr).x ) >= 0) ) ||
							( ( (int) ( (*point_ptr).y ) < buffer.voxels_size.y) && ( (int) ( (*point_ptr).y ) >= 0) ) ||
							( ( (int) ( (*point_ptr).z ) < buffer.voxels_size.z) && ( (int) ( (*point_ptr).z ) >= 0) )
							)//Not sure whether this long condition is necessary since destination indices are calculated based on rolling buffer
					{
						int dest_x = ( (int) ( (*point_ptr).x ) + buffer.origin_GRID.x ) % buffer.voxels_size.x;
						int dest_y = ( (int) ( (*point_ptr).y ) + buffer.origin_GRID.y ) % buffer.voxels_size.y;
						int dest_z = ( (int) ( (*point_ptr).z ) + buffer.origin_GRID.z ) % buffer.voxels_size.z;

						short2 *dest_tsdf_index = first_point_pointer + ( buffer.voxels_size.y * buffer.voxels_size.x * dest_z + ( buffer.voxels_size.x * dest_y ) ) + dest_x;
						(*dest_tsdf_index).x = (*tsdfvalue_ptr).x * divisor;
						(*dest_tsdf_index).y = 1.0;
					}
				}
				else
					return;
			}

			void
			pushCloudAsSliceGPU (const PtrStep<short2>& volume, kinfu::gpu::DeviceArray<PointType> cloud_gpu, const kinfu::gpu::tsdf_buffer* buffer)
			{
				// Because every point is 8 floats.
				// X Y Z 0
				// I 0 0 0
				int number_of_points = cloud_gpu.size () / 2;

				if(number_of_points == 0)
					return;

				const int DIVISOR = 32767;

				const short2 *first_point_pointer_const = &(volume.ptr (0)[0]);
				short2  *first_point_pointer = const_cast<short2*>(first_point_pointer_const);

				const int BlockX = 512;
				const int BlockY = 1;
				const int BlockSize = BlockX * BlockY;
				int numBlocks = divUp (number_of_points, BlockSize);
				dim3 threadsPerBlock (BlockX, BlockY);

				loadTsdfValueKernel <<< numBlocks, threadsPerBlock >>> (first_point_pointer, cloud_gpu.ptr(), number_of_points, DIVISOR, *buffer);
				cudaSafeCall ( cudaGetLastError () );
				cudaSafeCall ( cudaDeviceSynchronize () );
			}

	} /*namespace device*/
} /*namespace pcl*/

