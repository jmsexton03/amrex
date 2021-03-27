#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_Arena.H>
#include <AMReX_ParmParse.H>
#include <AMReX_FabArrayBase.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_NonLocalBC.H>

#include <AMReX_BArena.H>
#include <AMReX_CArena.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
      BL_PROFILE("main");
      int n_cell=256;
      {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
        pp.query("n_cell",n_cell);

      }
        IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
        IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
        Box domain(dom_lo, dom_hi);
        BaseFab<int> localtouch(The_Cpu_Arena()), remotetouch(The_Cpu_Arena());
	bool check_local, check_remote, m_threadsafe_loc, m_threadsafe_rcv;
	localtouch.resize(domain);
	localtouch.setVal<RunOn::Host>(0);
	remotetouch.resize(domain);
	remotetouch.setVal<RunOn::Host>(2);
	BL_PROFILE_VAR("CPU-max-1",max1);
	check_local = m_threadsafe_loc = localtouch.max<RunOn::Host>() <= 1;
	BL_PROFILE_VAR_STOP(max1);
	  BL_PROFILE_VAR("CPU-max-2",max2);
	check_remote = m_threadsafe_rcv = remotetouch.max<RunOn::Host>() <= 1;
	BL_PROFILE_VAR_STOP(max1);
	amrex::Print()<<"Found these max values: "<<check_local<<"\t"<<check_remote<<std::endl;
    }
    amrex::Finalize();
}

