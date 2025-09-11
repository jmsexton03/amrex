
#include "AMReX_ParallelDescriptor.H"
#include <AMReX_VisMF.H>
#include <AMReX_AsyncOut.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_FPC.H>
#include <AMReX_FabArrayUtility.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBFabFactory.H>
#endif

#include <fstream>
#include <iomanip>

namespace amrex {

void WriteJobInfo (const std::string& dir,
                   amrex::Real cpu_time,
                   const std::string& code_name,
                   const std::string& filename,
                   const BoxArray* grids,
                   const Geometry* geom,
                   const Vector<int>* lo_bc,
                   const Vector<int>* hi_bc,
                   const std::vector<std::string>* bc_names,
                   const std::string& job_name,
                   const std::string& inputs_name)
{
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream jobInfoFile;
        std::string FullPathJobInfoFile = dir + "/" + filename;
        jobInfoFile.open(FullPathJobInfoFile.c_str(), std::ios::out);

        std::string app_name = code_name;
        if (app_name.empty()) {
            app_name = get_application_name();
        }

        WriteJobHeader(jobInfoFile, app_name, job_name, inputs_name, 0.0, cpu_time);
        WriteOutputInfo(jobInfoFile);
        WriteDeviceMemoryInfo(jobInfoFile);
        WriteBuildInfo(jobInfoFile);

        if (grids && geom) {
            Vector<BoxArray> grids_vec = {*grids};
            Vector<Geometry> geom_vec = {*geom};
            Vector<IntVect> ref_ratio_vec;
            WriteGridInfo(jobInfoFile, grids_vec, geom_vec, ref_ratio_vec, 0);
        }

        if (lo_bc && hi_bc && bc_names) {
            WriteBoundaryInfo(jobInfoFile, *lo_bc, *hi_bc, *bc_names);
        }

        WriteRuntimeParameters(jobInfoFile);

        jobInfoFile.close();
    }
}

void WriteMultiLevelJobInfo (const std::string& dir,
                             amrex::Real cpu_time,
                             const std::string& code_name,
                             const std::string& filename,
                             const Vector<BoxArray>* grids,
                             const Vector<Geometry>* geom,
                             const Vector<IntVect>* ref_ratio,
                             int max_level,
                             const Vector<int>* blocking_factor,
                             const Vector<IntVect>* max_grid_size,
                             const Vector<int>* n_error_buf,
                             const Vector<int>* regrid_int,
                             const Vector<int>* lo_bc,
                             const Vector<int>* hi_bc,
                             const std::vector<std::string>* bc_names,
                             const std::string& job_name,
                             const std::string& inputs_name)
{
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream jobInfoFile;
        std::string FullPathJobInfoFile = dir + "/" + filename;
        jobInfoFile.open(FullPathJobInfoFile.c_str(), std::ios::out);

        std::string app_name = code_name;
        if (app_name.empty()) {
            app_name = get_application_name();
        }

        WriteJobHeader(jobInfoFile, app_name, job_name, inputs_name, 0.0, cpu_time);
        WriteOutputInfo(jobInfoFile);
        WriteDeviceMemoryInfo(jobInfoFile);
        WriteBuildInfo(jobInfoFile);

        if (grids && geom && ref_ratio) {
            WriteGridInfo(jobInfoFile, *grids, *geom, *ref_ratio, max_level,
                           blocking_factor, max_grid_size, n_error_buf, regrid_int);
        }

        if (lo_bc && hi_bc && bc_names) {
            WriteBoundaryInfo(jobInfoFile, *lo_bc, *hi_bc, *bc_names);
        }

        WriteRuntimeParameters(jobInfoFile);

        jobInfoFile.close();
    }
}
