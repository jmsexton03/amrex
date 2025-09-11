#ifndef AMREX_PLOTFILEUTIL_H_
#define AMREX_PLOTFILEUTIL_H_
#include <AMReX_Config.H>

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileDataImpl.H>
#include <AMReX_Vector.H>

#ifdef AMREX_USE_HDF5
#include <AMReX_PlotFileUtilHDF5.H>
#endif

#include <ostream>
#include <string>
#include <memory>

namespace amrex
{
void WriteJobInfo (const std::string& dir,
                   amrex::Real cpu_time,
                   const std::string& code_name = "",
                   const std::string& filename = "job_info",
                   const BoxArray* grids = nullptr,
                   const Geometry* geom = nullptr,
                   const Vector<int>* lo_bc = nullptr,
                   const Vector<int>* hi_bc = nullptr,
                   const std::vector<std::string>* bc_names = nullptr,
                   const std::string& job_name = "",
                   const std::string& inputs_name = "");

void WriteMultiLevelJobInfo (const std::string& dir,
                             amrex::Real cpu_time,
                             const std::string& code_name = "",
                             const std::string& filename = "job_info",
                             const Vector<BoxArray>* grids = nullptr,
                             const Vector<Geometry>* geom = nullptr,
                             const Vector<IntVect>* ref_ratio = nullptr,
                             int max_level = 0,
                             const Vector<int>* blocking_factor = nullptr,
                             const Vector<IntVect>* max_grid_size = nullptr,
                             const Vector<int>* n_error_buf = nullptr,
                             const Vector<int>* regrid_int = nullptr,
                             const Vector<int>* lo_bc = nullptr,
                             const Vector<int>* hi_bc = nullptr,
                             const std::vector<std::string>* bc_names = nullptr,
                             const std::string& job_name = "",
                             const std::string& inputs_name = "");

    // helper class for reading plotfile
    class JobInfoFileData
    {
    public:
      // Helper function to get application name from build info
std::string get_application_name() {
    const char* build_name = amrex::buildInfoGetBuildGitRepoName();
    if (build_name && strlen(build_name) > 0) {
        return std::string(build_name);
    }
    return "AMReX Application";
}

// Helper function to write job execution information
void WriteJobHeader(std::ofstream& jobInfoFile,
                      const std::string& app_name,
                      const std::string& job_name = "",
                      const std::string& inputs_name = "",
                      double io_time = 0.0,
                      double cpu_time = 0.0,
                      const amrex::IntVect& hydro_tile_size = amrex::IntVect::TheZeroVector(),
                      const amrex::IntVect& farraybox_tile_size = amrex::IntVect::TheZeroVector()) {
    std::string PrettyLine = std::string(78, '=') + "\n";

    // Job information header
    jobInfoFile << PrettyLine;
    jobInfoFile << " " << app_name << " Job Information\n";
    jobInfoFile << PrettyLine;

    // Use job_name from build info if not provided
    std::string actual_job_name = job_name;
    if (actual_job_name.empty()) {
        actual_job_name = get_application_name();
    }
    if (!actual_job_name.empty() && actual_job_name != "AMReX Application") {
        jobInfoFile << "job name: " << actual_job_name << "\n\n";
    }

    // Use "unknown" if no inputs name provided
    std::string actual_inputs_name = inputs_name;
    if (actual_inputs_name.empty()) {
        actual_inputs_name = "unknown";
    }
    jobInfoFile << "inputs file: " << actual_inputs_name << "\n\n";

    jobInfoFile << "number of MPI processes: " << amrex::ParallelDescriptor::NProcs() << "\n";
#ifdef AMREX_USE_OMP
    jobInfoFile << "number of threads:       " << omp_get_max_threads() << "\n";
    jobInfoFile << "tile size: ";
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        jobInfoFile << amrex::FabArrayBase::mfiter_tile_size[d] << " ";
    }
    jobInfoFile << "\n";

    // Additional tile size information if provided
    if (hydro_tile_size != amrex::IntVect::TheZeroVector()) {
        jobInfoFile << "hydro tile size: ";
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            jobInfoFile << hydro_tile_size[d] << " ";
        }
        jobInfoFile << "\n";
    }

    if (farraybox_tile_size != amrex::IntVect::TheZeroVector()) {
        jobInfoFile << "farraybox tile size: ";
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            jobInfoFile << farraybox_tile_size[d] << " ";
        }
        jobInfoFile << "\n";
    }
#endif
    jobInfoFile << "\n";

#ifdef AMREX_USE_GPU
    jobInfoFile << "GPU time used since start of simulation (GPU-hours): " <<
        cpu_time/3600.0 << "\n";
#else
    jobInfoFile << "CPU time used since start of simulation (CPU-hours): " <<
        cpu_time/3600.0 << "\n";
#endif

    // Additional I/O timing if provided
    if (io_time > 0.0) {
        jobInfoFile << "I/O time used since start of simulation (hours): " <<
            io_time/3600.0 << "\n";
    }

    jobInfoFile << "\n";
}

// Helper function to write output information
void WriteOutputInfo(std::ofstream& jobInfoFile) {
    std::string PrettyLine = std::string(78, '=') + "\n";

    jobInfoFile << PrettyLine;
    jobInfoFile << " Output Information\n";
    jobInfoFile << PrettyLine;

    const std::time_t now = time(nullptr);
    jobInfoFile << "output date / time: "
                << std::put_time(std::localtime(&now), "%c\n") << "\n";
    jobInfoFile << "output dir:         " << amrex::FileSystem::CurrentPath() << "\n";
    jobInfoFile << "\n\n";
}

// Helper function to write device and memory information (AMReX general)
void WriteDeviceMemoryInfo(std::ofstream& jobInfoFile) {
    std::string PrettyLine = std::string(78, '=') + "\n";

    jobInfoFile << PrettyLine;
    jobInfoFile << " Device and Memory Information\n";
    jobInfoFile << PrettyLine;

#ifdef AMREX_USE_GPU
    // GPU Backend type
#if defined(AMREX_USE_CUDA)
    jobInfoFile << "GPU Backend: CUDA\n";
#elif defined(AMREX_USE_HIP)
    jobInfoFile << "GPU Backend: HIP\n";
#elif defined(AMREX_USE_SYCL)
    jobInfoFile << "GPU Backend: SYCL\n";
#endif

    jobInfoFile << "Device ID: " << amrex::Gpu::Device::deviceId() << "\n";
    jobInfoFile << "Number of devices used: " << amrex::Gpu::Device::numDevicesUsed() << "\n";
    jobInfoFile << "Number of device partners: " << amrex::Gpu::Device::numDevicePartners() << "\n";
    jobInfoFile << "Max GPU streams: " << amrex::Gpu::Device::numGpuStreams() << "\n";

    // Device properties
    auto& prop = amrex::Gpu::Device::deviceProperties();
    jobInfoFile << "Device name: " << prop.name << "\n";
    jobInfoFile << "Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    jobInfoFile << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
    jobInfoFile << "Multiprocessor count: " << prop.multiProcessorCount << "\n";
    jobInfoFile << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    jobInfoFile << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
    jobInfoFile << "Warp size: " << prop.warpSize << "\n";

    // Backend-specific properties
#ifdef AMREX_USE_CUDA
    jobInfoFile << "Compute capability: " << prop.major << "." << prop.minor << "\n";
#endif

#ifdef AMREX_USE_HIP
    jobInfoFile << "GCN architecture: " << prop.gcnArch << "\n";
    jobInfoFile << "Max shared memory per multiprocessor: " << prop.maxSharedMemoryPerMultiProcessor << " bytes\n";
    jobInfoFile << "Memory clock rate: " << prop.memoryClockRate << " kHz\n";
    jobInfoFile << "Memory bus width: " << prop.memoryBusWidth << " bits\n";
#endif

#ifdef AMREX_USE_SYCL
    jobInfoFile << "Device vendor: " << prop.vendor << "\n";
    jobInfoFile << "Max memory allocation size: " << prop.maxMemAllocSize / (1024*1024) << " MB\n";
    jobInfoFile << "Managed memory support: " << (prop.managedMemory ? "Yes" : "No") << "\n";
    jobInfoFile << "Concurrent managed access: " << (prop.concurrentManagedAccess ? "Yes" : "No") << "\n";

// Intel GPU specific information (if available)
#if defined(__INTEL_LLVM_COMPILER)
    auto& d = *amrex::Gpu::Device::sycl_device;
    if (d.has(sycl::aspect::ext_intel_gpu_eu_count)) {
        jobInfoFile << "Intel GPU Execution Units: " <<
            d.get_info<sycl::ext::intel::info::device::gpu_eu_count>() << "\n";
    }
    if (d.has(sycl::aspect::ext_intel_gpu_slices)) {
        jobInfoFile << "Intel GPU Slices: " <<
            d.get_info<sycl::ext::intel::info::device::gpu_slices>() << "\n";
    }
    if (d.has(sycl::aspect::ext_intel_max_mem_bandwidth)) {
        jobInfoFile << "Intel GPU Max Memory Bandwidth: " <<
            d.get_info<sycl::ext::intel::info::device::max_mem_bandwidth>() / (1024*1024*1024) << " GB/s\n";
    }
#endif
#endif

    // Current memory usage
    jobInfoFile << "Free memory available: " <<
        amrex::Gpu::Device::freeMemAvailable() / (1024*1024) << " MB\n";

    jobInfoFile << "\n";
#endif

    // Memory usage information (works for both CPU and GPU builds)
    jobInfoFile << "AMReX Space Dimension: " << AMREX_SPACEDIM << "\n";
#ifdef AMREX_USE_FLOAT
    jobInfoFile << "AMReX Precision: Single (float)\n";
#else
    jobInfoFile << "AMReX Precision: Double\n";
#endif

    // MultiFab memory usage (local to this process, no MPI needed)
    long long multifab_mem = amrex::TotalBytesAllocatedInFabs();
    if (multifab_mem > 0) {
        jobInfoFile << "MultiFab memory allocated: " << multifab_mem / (1024*1024) << " MB\n";
    }

    // Arena information (if available without MPI)
    jobInfoFile << "Default Arena initialized: " << (amrex::The_Arena() != nullptr ? "Yes" : "No") << "\n";
#ifdef AMREX_USE_GPU
    jobInfoFile << "Device Arena initialized: " << (amrex::The_Device_Arena() != nullptr ? "Yes" : "No") << "\n";
    jobInfoFile << "Managed Arena initialized: " << (amrex::The_Managed_Arena() != nullptr ? "Yes" : "No") << "\n";
    jobInfoFile << "Pinned Arena initialized: " << (amrex::The_Pinned_Arena() != nullptr ? "Yes" : "No") << "\n";
#endif

    jobInfoFile << "\n\n";
}

// Helper function to write build information (AMReX general) - Changed to accept std::ostream&
void WriteBuildInfo(std::ostream& jobInfoFile) {
    std::string PrettyLine = std::string(78, '=') + "\n";

    jobInfoFile << PrettyLine;
    jobInfoFile << " Build Information\n";
    jobInfoFile << PrettyLine;

    jobInfoFile << "build date:    " << amrex::buildInfoGetBuildDate() << "\n";
    jobInfoFile << "build machine: " << amrex::buildInfoGetBuildMachine() << "\n";
    jobInfoFile << "build dir:     " << amrex::buildInfoGetBuildDir() << "\n";
    jobInfoFile << "AMReX dir:     " << amrex::buildInfoGetAMReXDir() << "\n";

    jobInfoFile << "\n";

    jobInfoFile << "COMP:          " << amrex::buildInfoGetComp() << "\n";
    jobInfoFile << "COMP version:  " << amrex::buildInfoGetCompVersion() << "\n";

    jobInfoFile << "\n";

#ifdef AMREX_USE_CUDA
    jobInfoFile << "CUDA version:  " << buildInfoGetCUDAVersion() << "\n";
    jobInfoFile << "\n";
#endif

    jobInfoFile << "C++ compiler:  " << amrex::buildInfoGetCXXName() << "\n";
    jobInfoFile << "C++ flags:     " << amrex::buildInfoGetCXXFlags() << "\n";

    jobInfoFile << "\n";

    jobInfoFile << "Fortran comp:  " << amrex::buildInfoGetFName() << "\n";
    jobInfoFile << "Fortran flags: " << amrex::buildInfoGetFFlags() << "\n";

    jobInfoFile << "\n";

    jobInfoFile << "Link flags:    " << amrex::buildInfoGetLinkFlags() << "\n";
    jobInfoFile << "Libraries:     " << amrex::buildInfoGetLibraries() << "\n";

    jobInfoFile << "\n";

    for (int n = 1; n <= amrex::buildInfoGetNumModules(); n++) {
        jobInfoFile << amrex::buildInfoGetModuleName(n) << ": " << amrex::buildInfoGetModuleVal(n) << "\n";
    }

    jobInfoFile << "\n";

    // Git information for all repositories
    int git_repo = 1;
    while (true) {
        const char* git_name = amrex::buildInfoGetGitRepoName(git_repo);
        const char* git_hash = amrex::buildInfoGetGitHash(git_repo);

        if (git_name == nullptr || strlen(git_name) == 0 ||
            git_hash == nullptr || strlen(git_hash) == 0) {
            break;
        }

        jobInfoFile << std::setw(15) << std::left << git_name
                    << " git describe: " << git_hash << "\n";
        git_repo++;
    }

    jobInfoFile << "\n\n";
}

// Helper function to write grid information
void WriteGridInfo(std::ofstream& jobInfoFile,
                     const amrex::Vector<amrex::BoxArray>& grids,
                     const amrex::Vector<amrex::Geometry>& geom,
                     const amrex::Vector<amrex::IntVect>& ref_ratio,
                     int max_level,
                     const amrex::Vector<int>* blocking_factor = nullptr,
                     const amrex::Vector<amrex::IntVect>* max_grid_size = nullptr,
                     const amrex::Vector<int>* n_error_buf = nullptr,
                     const amrex::Vector<int>* regrid_int = nullptr) {
    std::string PrettyLine = std::string(78, '=') + "\n";

    jobInfoFile << PrettyLine;
    jobInfoFile << " Grid Information\n";
    jobInfoFile << PrettyLine;

    for (int i = 0; i <= max_level; i++) {
        jobInfoFile << " level: " << i << "\n";
        jobInfoFile << "   number of boxes = " << grids[i].size() << "\n";
        jobInfoFile << "   maximum zones   = ";
        for (int n = 0; n < AMREX_SPACEDIM; n++) {
            jobInfoFile << geom[i].Domain().length(n) << " ";
        }
        jobInfoFile << "\n\n";
    }

    jobInfoFile << " Domain geometry info\n";

    jobInfoFile << "     geometry.is_periodic: ";
    for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
        jobInfoFile << geom[0].isPeriodic(idir) << " ";
    }
    jobInfoFile << "\n";

    jobInfoFile << "     geometry.coord_sys:   " << geom[0].Coord() << "\n";

    jobInfoFile << "     geometry.prob_lo:     ";
    for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
        jobInfoFile << geom[0].ProbLo(idir) << " ";
    }
    jobInfoFile << "\n";

    jobInfoFile << "     geometry.prob_hi:     ";
    for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
        jobInfoFile << geom[0].ProbHi(idir) << " ";
    }
    jobInfoFile << "\n";

    jobInfoFile << "     amr.n_cell:           ";
    const int* domain_lo = geom[0].Domain().loVect();
    const int* domain_hi = geom[0].Domain().hiVect();
    for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
        jobInfoFile << domain_hi[idir] - domain_lo[idir] + 1 << " ";
    }
    jobInfoFile << "\n";

    jobInfoFile << "     amr.max_level:        " << max_level << "\n";

    if (max_level > 0) {
        jobInfoFile << "     amr.ref_ratio:        ";
        for (int lev = 0; lev < max_level; lev++) {
            jobInfoFile << ref_ratio[lev][0] << " ";
        }
        jobInfoFile << "\n";
    }

    // Optional AMR parameters
    if (n_error_buf) {
        jobInfoFile << "     amr.n_error_buf:      ";
        for (int lev = 0; lev < max_level; lev++) {
            jobInfoFile << (*n_error_buf)[lev] << " ";
        }
        jobInfoFile << "\n";
    }

    if (regrid_int) {
        jobInfoFile << "     amr.regrid_int:       ";
        for (int lev = 0; lev < max_level; lev++) {
            jobInfoFile << (*regrid_int)[lev] << " ";
        }
        jobInfoFile << "\n";
    }

    if (blocking_factor) {
        jobInfoFile << "     amr.blocking_factor:  ";
        for (int lev = 0; lev <= max_level; lev++) {
            jobInfoFile << (*blocking_factor)[lev] << " ";
        }
        jobInfoFile << "\n";
    }

    if (max_grid_size) {
        jobInfoFile << "     amr.max_grid_size:    ";
        for (int lev = 0; lev <= max_level; lev++) {
            jobInfoFile << (*max_grid_size)[lev][0] << " ";
        }
        jobInfoFile << "\n";
    }

    jobInfoFile << "\n\n";
}

// Helper function to write boundary condition information
void WriteBoundaryInfo(std::ofstream& jobInfoFile,
                         const amrex::Vector<int>& lo_bc,
                         const amrex::Vector<int>& hi_bc,
                         const std::vector<std::string>& bc_names) {
    std::string PrettyLine = std::string(78, '=') + "\n";

    jobInfoFile << " Boundary conditions\n";
    jobInfoFile << "   -x: " << bc_names[lo_bc[0]] << "\n";
    jobInfoFile << "   +x: " << bc_names[hi_bc[0]] << "\n";
    if (AMREX_SPACEDIM >= 2) {
        jobInfoFile << "   -y: " << bc_names[lo_bc[1]] << "\n";
        jobInfoFile << "   +y: " << bc_names[hi_bc[1]] << "\n";
    }
    if (AMREX_SPACEDIM == 3) {
        jobInfoFile << "   -z: " << bc_names[lo_bc[2]] << "\n";
        jobInfoFile << "   +z: " << bc_names[hi_bc[2]] << "\n";
    }
    jobInfoFile << "\n\n";
}

// Helper function to write species information (for physics codes)
void WriteSpeciesInfo(std::ofstream& jobInfoFile,
                        int num_species,
                        const std::vector<std::string>& species_names,
                        const std::vector<double>& atomic_mass,
                        const std::vector<double>& atomic_number) {
    std::string PrettyLine = std::string(78, '=') + "\n";
    std::string OtherLine = std::string(78, '-') + "\n";
    std::string SkipSpace = std::string(8, ' ');

    jobInfoFile << PrettyLine;
    jobInfoFile << " Species Information\n";
    jobInfoFile << PrettyLine;

    int mlen = 20;
    jobInfoFile <<
        std::setw(6) << "index" << SkipSpace <<
        std::setw(mlen+1) << "name" << SkipSpace <<
        std::setw(7) << "A" << SkipSpace <<
        std::setw(7) << "Z" << "\n";
    jobInfoFile << OtherLine;

    for (int i = 0; i < num_species; i++) {
        jobInfoFile <<
            std::setw(6) << i << SkipSpace <<
            std::setw(mlen+1) << std::setfill(' ') << species_names[i] << SkipSpace <<
            std::setw(7) << atomic_mass[i] << SkipSpace <<
            std::setw(7) << atomic_number[i] << "\n";
    }
    jobInfoFile << "\n\n";
}

// Helper function to write runtime parameters
void WriteRuntimeParameters(std::ofstream& jobInfoFile) {
    std::string PrettyLine = std::string(78, '=') + "\n";

    jobInfoFile << PrettyLine;
    jobInfoFile << " Inputs File Parameters\n";
    jobInfoFile << PrettyLine;

    amrex::ParmParse::prettyPrintTable(jobInfoFile);
}

      //    private:

    };
}

#endif
