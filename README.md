# nbody


## Nbody sample
An N-body simulation is a simulation of a dynamical system of particles, usually under the influence of physical forces, such as gravity. This nbody sample code is implemented using C++ and DPC++ language for Intel CPU and GPU.

## Purpose
Nbody sample code simulates 32000 particles and for ten integration steps. Each particle's position, velocity and acceleration parameters are dependent on other (N-1) particles. This algorithm is highly data parallel and a perfect candidate to offload to GPU. The code demonstrates how to deal with multiple device kernels, which can be enqueued into a DPC++ queue for execution and how to handle parallel reductions.

## Key Implementation Details
The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.
## The Learning experience(new app)
##Good
Good amount of examples
Very good way to get started with Samples
##Bad
Learning curve is steep
Devcloud is slow at times

## Extending the Sample
Adding a superficial 4th dimension for the N-body Simulation

## Modifying the particle.hpp

struct Particle {
 public:
  Particle() : pos{}, vel{}, acc{}, mass{} {};
  RealType pos[4];
  RealType vel[4];
  RealType acc[4];
  RealType mass;
};

## Modifying the Gsimulation.cpp

for (int i = 0; i < get_npart(); ++i) {
    particles_[i].pos[0] = unif_d(gen);
    particles_[i].pos[1] = unif_d(gen);
    particles_[i].pos[2] = unif_d(gen);
    particles_[i].pos[3] = unif_d(gen);
  }
  

for (int i = 0; i < get_npart(); ++i) {
    particles_[i].vel[0] = unif_d(gen) * 1.0e-3f;
    particles_[i].vel[1] = unif_d(gen) * 1.0e-3f;
    particles_[i].vel[2] = unif_d(gen) * 1.0e-3f;
    particles_[i].vel[3] = unif_d(gen) * 1.0e-3f;
  }
  
  
// Looping across integration steps
  for (int s = 1; s <= nsteps; ++s) {
    dpc_common::TimeInterval ts0;
    // Submitting first kernel to device which computes acceleration of all
    // particles
    q.submit([&](handler& h) {
       auto p = pbuf.get_access(h);
       h.parallel_for(ndrange, [=](nd_item<1> it) {
	 auto i = it.get_global_id();
         RealType acc0 = p[i].acc[0];
         RealType acc1 = p[i].acc[1];
         RealType acc2 = p[i].acc[2];
         RealType acc3 = p[i].acc[3];
         for (int j = 0; j < n; j++) {
           RealType dx, dy, dz, dm;
           RealType distance_sqr = 0.0f;
           RealType distance_inv = 0.0f;

           dx = p[j].pos[0] - p[i].pos[0];  // 1flop
           dy = p[j].pos[1] - p[i].pos[1];  // 1flop
           dz = p[j].pos[2] - p[i].pos[2];  // 1flop
           dm = p[j].pos[3] - p[i].pos[3];

           distance_sqr =
               dx * dx + dy * dy + dz * dz + dm * dm + kSofteningSquared;  // 6flops
           distance_inv = 1.0f / sycl::sqrt(distance_sqr);       // 1div+1sqrt

           acc0 += dx * kG * p[j].mass * distance_inv * distance_inv *
                   distance_inv;  // 6flops
           acc1 += dy * kG * p[j].mass * distance_inv * distance_inv *
                   distance_inv;  // 6flops
           acc2 += dz * kG * p[j].mass * distance_inv * distance_inv *
                   distance_inv;  // 6flops
           acc3 += dm * kG * p[j].mass * distance_inv * distance_inv *
                   distance_inv; 
         }
         p[i].acc[0] = acc0;
         p[i].acc[1] = acc1;
         p[i].acc[2] = acc2;
         p[i].acc[3] = acc3;
       });
     }).wait_and_throw();
     
     
   ## Running the Sample
### To run the program in Devcloud
unzip the zip file
### mkdir <dir-name>
### Get inside the directory
### cd <dir-name>
###Then use Cmake
### cmake ..

### u:~/oneAPI-samples/DirectProgramming/DPC++/N-BodyMethods/Nbody/fried$ cmake ..
##
### -- The C compiler identification is GNU 7.4.0
-- The CXX compiler identification is Clang 12.0.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /glob/development-tools/versions/oneapi/gold/inteloneapi/compiler/2021.1.2/linux/bin/dpcpp
-- Check for working CXX compiler: /glob/development-tools/versions/oneapi/gold/inteloneapi/compiler/2021.1.2/linux/bin/dpcpp -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
### -- Build files have been written to: /home/u/oneAPI-samples/DirectProgramming/DPC++/N-BodyMethods/Nbody/fried

##

### Building the file
### make
### Scanning dependencies of target nbody
[ 33%] Building CXX object src/CMakeFiles/nbody.dir/GSimulation.cpp.o
[ 66%] Building CXX object src/CMakeFiles/nbody.dir/main.cpp.o
[100%] Linking CXX executable nbody
### [100%] Built target nbody

### running the application

### make run

Scanning dependencies of target run
===============================
 Initialize Gravity Simulation
 nPart = 32000; nSteps = 20; dt = 0.1
------------------------------------------------
 s       dt      kenergy     time (s)    GFLOPS      
------------------------------------------------
 1       0.1     5892.8      1.8653      15.92       
 2       0.2     28969       0.076232    389.56      
 3       0.3     69675       0.06694     443.63      
 4       0.4     1.282e+05   0.067076    442.73      
 5       0.5     2.0482e+05  0.069259    428.77      
 6       0.6     2.9991e+05  0.068376    434.31      
 7       0.7     4.1392e+05  0.067151    442.24      
 8       0.8     5.4743e+05  0.069639    426.43      
 9       0.9     7.011e+05   0.069146    429.48      
 10      1       8.7574e+05  0.070268    422.62      
 11      1.1     1.0723e+06  0.066957    443.52      
 12      1.2     1.2917e+06  0.06686     444.16      
 13      1.3     1.5354e+06  0.066728    445.04      
 14      1.4     1.8047e+06  0.067011    443.16      
 15      1.5     2.1011e+06  0.066848    444.24      
 16      1.6     2.4267e+06  0.067243    441.63      
 17      1.7     2.7834e+06  0.069707    426.02      
 18      1.8     3.1737e+06  0.066968    443.44      
 19      1.9     3.6005e+06  0.067454    440.25      
 20      2       4.0669e+06  0.066916    443.79      

# Total Time (s)     : 3.1648
# Average Performance : 438.08 +- 7.5355
===============================
Built target runimension


## Summary
# Increased G-Flops
# Added 4th dimension
# Simulation starting point 32000
