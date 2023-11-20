#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>
#include <TerrainFittedPC.H>

using namespace amrex;

static constexpr int NSR = 6;
static constexpr int NSI = 1;
static constexpr int NAR = 1;
static constexpr int NAI = 1;

int num_runtime_real = 0;
int num_runtime_int = 0;

bool remove_negative = true;

bool zero_center = false;

void get_position_unit_cell(Real* r, const IntVect& nppc, int i_part)
{
    int nx = nppc[0];
#if AMREX_SPACEDIM > 1
    int ny = nppc[1];
#else
    int ny = 1;
#endif
#if AMREX_SPACEDIM > 2
    int nz = nppc[2];
#else
    int nz = 1;
#endif

    int ix_part = i_part/(ny * nz);
    int iy_part = (i_part % (ny * nz)) % ny;
    int iz_part = (i_part % (ny * nz)) / ny;
    if(!zero_center) {
    r[0] = (0.5+ix_part)/nx;
    r[1] = (0.5+iy_part)/ny;
    r[2] = (0.5+iz_part)/nz;
    } else {
    Abort("assumptions on problo");
    }
}

struct TestParams
{
    IntVect size;
    int max_grid_size;
    int num_ppc;
    int is_periodic;
    IntVect move_dir;
    int do_random;
    int nsteps;
    int nlevs;
    int do_regrid;
    int sort;
    int test_level_lost = 0;
};

void testRedistribute();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    amrex::Print() << "Running redistribute test \n";
    testRedistribute();

    amrex::Finalize();
}

void get_test_params(TestParams& params, const std::string& prefix)
{
    ParmParse pp(prefix);
    pp.get("size", params.size);
    pp.get("max_grid_size", params.max_grid_size);
    pp.get("num_ppc", params.num_ppc);
    pp.get("is_periodic", params.is_periodic);
    pp.get("move_dir", params.move_dir);
    pp.get("do_random", params.do_random);
    pp.get("nsteps", params.nsteps);
    pp.get("nlevs", params.nlevs);
    pp.get("do_regrid", params.do_regrid);
    pp.query("test_level_lost", params.test_level_lost);
    pp.query("num_runtime_real", num_runtime_real);
    pp.query("num_runtime_int", num_runtime_int);
    pp.query("remove_negative", remove_negative);
    pp.query("zero_center", zero_center);

    params.sort = 0;
    pp.query("sort", params.sort);
}

void testRedistribute ()
{
    BL_PROFILE("testRedistribute");
    TestParams params;
    get_test_params(params, "redistribute");

    int is_per[] = {AMREX_D_DECL(params.is_periodic,
                                 params.is_periodic,
                                 params.is_periodic)};

    Vector<IntVect> rr(params.nlevs-1);
    for (int lev = 1; lev < params.nlevs; lev++) {
        rr[lev-1] = IntVect(AMREX_D_DECL(2,2,2));
    }

    RealBox real_box;
    for (int n = 0; n < BL_SPACEDIM; n++)
    {
        if(zero_center)
            real_box.setLo(n, -params.size[n]);
        else
            real_box.setLo(n, 0.0);
        real_box.setHi(n, params.size[n]);
    }

    IntVect domain_lo(zero_center ? AMREX_D_DECL(-params.size[0]+1,-params.size[1]+1,-params.size[2]+1): AMREX_D_DECL(0, 0, 0));
    IntVect domain_hi(AMREX_D_DECL(params.size[0]-1,params.size[1]-1,params.size[2]-1));
    const Box base_domain(domain_lo, domain_hi);

    Vector<Geometry> geom(params.nlevs);
    geom[0].define(base_domain, &real_box, CoordSys::cartesian, is_per);
    for (int lev = 1; lev < params.nlevs; lev++) {
        geom[lev].define(amrex::refine(geom[lev-1].Domain(), rr[lev-1]),
                         &real_box, CoordSys::cartesian, is_per);
    }

    Vector<BoxArray> ba(params.nlevs);
    Vector<DistributionMapping> dm(params.nlevs);
    IntVect lo(0);
    IntVect size = params.size;
    for (int lev = 0; lev < params.nlevs; ++lev)
    {
        ba[lev].define(Box(domain_lo, domain_hi));
        ba[lev].maxSize(params.max_grid_size);
        dm[lev].define(ba[lev]);
        lo += size/2;
        size *= 2;
    }

    TerrainFittedPC pc(geom[0], dm[0], ba[0]);

    IntVect nppc(params.num_ppc);

    amrex::Print() << "About to initialize particles \n";
    MultiFab a_z_height(pc.amrex::ParticleContainerBase::ParticleBoxArray(0),pc.amrex::ParticleContainerBase::ParticleDistributionMap(0),3,0);
    pc.InitHeight(a_z_height);
    MultiFab umac[3];
    umac[0].define(pc.amrex::ParticleContainerBase::ParticleBoxArray(0),pc.amrex::ParticleContainerBase::ParticleDistributionMap(0),3,0);
    umac[1].define(pc.amrex::ParticleContainerBase::ParticleBoxArray(0),pc.amrex::ParticleContainerBase::ParticleDistributionMap(0),3,0);
    umac[2].define(pc.amrex::ParticleContainerBase::ParticleBoxArray(0),pc.amrex::ParticleContainerBase::ParticleDistributionMap(0),3,0);
    umac[0].setVal(0.0);
    umac[1].setVal(0.0);
    umac[2].setVal(0.0);
    pc.InitUmac(&umac[0], 0, 1.0, a_z_height);
    pc.InitParticles(a_z_height);
    pc.WritePlotFile("plot", "particles");
    pc.WritePlotFile("plot0", "particles");
    pc.AdvectWithUmac(&umac[0], 0, 1.0, a_z_height);

    pc.WritePlotFile("plot", "particles");
    pc.WritePlotFile("plot1", "particles");
    pc.AdvectWithUmac(&umac[0], 0, 1.0, a_z_height);

    pc.WritePlotFile("plot2", "particles");
    pc.AdvectWithUmac(&umac[0], 0, 1.0, a_z_height);

    pc.WritePlotFile("plot3", "particles");
    pc.AdvectWithUmac(&umac[0], 0, 1.0, a_z_height);

    pc.WritePlotFile("plot4", "particles");
    pc.AdvectWithUmac(&umac[0], 0, 0.5, a_z_height);

    pc.WritePlotFile("plot5", "particles");
    pc.AdvectWithUmac(&umac[0], 0, -0.5, a_z_height);

    pc.WritePlotFile("plot6", "particles");
    pc.AdvectWithUmac(&umac[0], 0, -1.0, a_z_height);

    pc.WritePlotFile("plot7", "particles");
    pc.AdvectWithUmac(&umac[0], 0, -1.0, a_z_height);

    pc.AdvectWithUmac(&umac[0], 0, -1.0, a_z_height);

    pc.AdvectWithUmac(&umac[0], 0, -1.0, a_z_height);

    pc.WritePlotFile("plot8", "particles");

}
