__kernel void add_vectors(__global float* buffer, __global float* vec1, __global float* vec2, int vector_dimension) 
{
   float4 coord = (float4)(1.0, 2.0, 3.0, 1.0);
   if (get_global_id(0) < vector_dimension) {
      int id = get_global_id(0);
		  buffer[id] = vec1[id] + vec2[id] + coord.y;
    }
}
