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

    r[0] = (0.5+ix_part)/nx;
    r[1] = (0.5+iy_part)/ny;
    r[2] = (0.5+iz_part)/nz;
}
/*
    void negateEven ()
    {
        BL_PROFILE("TestParticleContainer::invalidateEven");

        for (int lev = 0; lev <= finestLevel(); ++lev)
        {
            auto& plev  = GetParticles(lev);
            for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
            {
                int gid = mfi.index();
                int tid = mfi.LocalTileIndex();
                auto& ptile = plev[std::make_pair(gid, tid)];
                auto& aos   = ptile.GetArrayOfStructs();
                ParticleType* pstruct = aos.data();
                const size_t np = aos.numParticles();
                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
                {
                    ParticleType& p = pstruct[i];
                    if (p.id() % 2 == 0) {
                        p.id() = -p.id();
                    }
                });
            }
        }
    }

    void checkAnswer () const
    {
        BL_PROFILE("TestParticleContainer::checkAnswer");

        AMREX_ALWAYS_ASSERT(OK());

        int num_rr = NumRuntimeRealComps();
        int num_ii = NumRuntimeIntComps();

        for (int lev = 0; lev <= finestLevel(); ++lev)
        {
            const auto& plev  = GetParticles(lev);
            for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
            {
                int gid = mfi.index();
                int tid = mfi.LocalTileIndex();
                const auto& ptile = plev.at(std::make_pair(gid, tid));
                const auto ptd = ptile.getConstParticleTileData();
                const size_t np = ptile.numParticles();

                AMREX_FOR_1D ( np, i,
                {
                    for (int j = 0; j < NSR; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_aos[i].rdata(j) == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < NSI; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_aos[i].idata(j) == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < NAR; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_rdata[j][i] == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < NAI; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_idata[j][i] == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < num_rr; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_runtime_rdata[j][i] == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < num_ii; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_runtime_idata[j][i] == ptd.m_aos[i].id());
                    }
                });
            }
        }
    }
};
*/
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
    for (int lev = 1; lev < params.nlevs; lev++)
        rr[lev-1] = IntVect(AMREX_D_DECL(2,2,2));

    RealBox real_box;
    for (int n = 0; n < BL_SPACEDIM; n++)
    {
        real_box.setLo(n, -params.size[n]);
        real_box.setHi(n, params.size[n]);
    }

    IntVect domain_lo(AMREX_D_DECL(-params.size[0]+1,-params.size[1]+1,-params.size[2]+1));
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

    pc.InitParticles();
    /*
    pc.checkAnswer();

    auto np_old = pc.TotalNumberOfParticles();

    if (params.sort) pc.SortParticlesByCell();

    for (int i = 0; i < params.nsteps; ++i)
    {
        pc.moveParticles(params.move_dir, params.do_random);
        if (!remove_negative) {
            auto old = pc.TotalNumberOfParticles();
            pc.negateEven();
            pc.RedistributeLocal(false);
            AMREX_ALWAYS_ASSERT(old == pc.TotalNumberOfParticles(false));
            pc.negateEven();
        }
        pc.RedistributeLocal();
        if (params.sort) pc.SortParticlesByCell();
        pc.checkAnswer();
    }

    if (params.do_regrid)
    {
        const int NProcs = ParallelDescriptor::NProcs();
        {
            for (int lev = 0; lev < params.nlevs; ++lev)
            {
                DistributionMapping new_dm;
                Vector<int> pmap;
                for (int i = 0; i < ba[lev].size(); ++i) pmap.push_back(i % NProcs);
                new_dm.define(pmap);
                pc.SetParticleDistributionMap(lev, new_dm);
            }
            if (!remove_negative) {
                auto old = pc.TotalNumberOfParticles();
                pc.negateEven();
                pc.RedistributeGlobal(false);
                AMREX_ALWAYS_ASSERT(old == pc.TotalNumberOfParticles(false));
                pc.negateEven();
            }
            pc.RedistributeGlobal();
            pc.checkAnswer();
        }

        {
            for (int lev = 0; lev < params.nlevs; ++lev)
            {
                DistributionMapping new_dm;
                Vector<int> pmap;
                for (int i = 0; i < ba[lev].size(); ++i) pmap.push_back((i+1) % NProcs);
                new_dm.define(pmap);
                pc.SetParticleDistributionMap(lev, new_dm);
            }
            if (!remove_negative) {
                auto old = pc.TotalNumberOfParticles();
                pc.negateEven();
                pc.RedistributeGlobal(false);
                AMREX_ALWAYS_ASSERT(old == pc.TotalNumberOfParticles(false));
                pc.negateEven();
            }
            pc.RedistributeGlobal();
            pc.checkAnswer();
        }

        if (params.test_level_lost) {
            AMREX_ALWAYS_ASSERT(params.nlevs > 2);
            auto np_before_level_lost = pc.TotalNumberOfParticles();
            Vector<BoxArray> new_ba = ba; new_ba.resize(ba.size()-1);
            Vector<DistributionMapping> new_dm = dm; new_dm.resize(dm.size()-1);
            Vector<Geometry> new_geom = geom; new_geom.resize(geom.size()-1);
            Vector<IntVect> new_rr = rr; new_rr.resize(rr.size()-1);
            pc.ParticleContainer::Define(new_geom, new_dm, new_ba, new_rr);
            pc.Redistribute();
            amrex::Print() << np_before_level_lost << "\n";
            amrex::Print() << pc.TotalNumberOfParticles() << "\n";
            AMREX_ALWAYS_ASSERT(np_before_level_lost == pc.TotalNumberOfParticles());
        }
    }

    if (geom[0].isAllPeriodic()) AMREX_ALWAYS_ASSERT(np_old == pc.TotalNumberOfParticles());

    // the way this test is set up, if we make it here we pass
    amrex::Print() << "pass \n";
    */
    pc.WritePlotFile("plot", "particles");
    //    pc.WriteAsciiFile("plot_ascii");
}
