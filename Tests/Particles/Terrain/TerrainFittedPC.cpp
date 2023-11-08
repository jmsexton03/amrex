#include "TerrainFittedPC.H"

#include <AMReX_TracerParticle_mod_K.H>

using namespace amrex;

static constexpr int NSR = 6;
static constexpr int NSI = 3;
static constexpr int NAR = 0;
static constexpr int NAI = 0;


void
TerrainFittedPC::
InitHeight (MultiFab& a_z_height)
{
    BL_PROFILE("TerrainFittedPC::InitHeight");

    const int lev = 0;
    const Real* dx = Geom(lev).CellSize();
    const Real* plo = Geom(lev).ProbLo();
    auto domain = this->amrex::ParticleContainerBase::Geom(0).Domain();
    auto probhi = this->amrex::ParticleContainerBase::Geom(0).ProbHi();
    auto problo = this->amrex::ParticleContainerBase::Geom(0).ProbLo();
    for(MFIter mfi(a_z_height); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.growntilebox();
        auto height_arr = a_z_height.array(mfi);
        //        Real r[3] = {0.5, 0.5, 0.5};  // this means place at cell center
        Real r[3] = {0.0, 0.0, 0.0};  // this means place at cell center
        const Real* dx = Geom(lev).CellSize();
        const Real* plo = Geom(lev).ProbLo();
        const Box tile_box101 = makeSlab(mfi.growntilebox(),1,0);
        const Box tile_box011 = makeSlab(mfi.growntilebox(),0,0);
        const Real pi=amrex::Math::pi<Real>();
        amrex::ParallelFor( tile_box101, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = problo[0]+(r[0]+i)*dx[0];
            Real cx  =problo[0]+0.5*(probhi[0]-problo[0]);
            Real cy  =problo[1]+0.5*(probhi[1]-problo[1]);
            Real xi, xo;
            Real r = (cx+cy)/2.0;
            if(x<cx)
                xi=sqrt(r*r-(cx-x)*(cx-x));
            else
                xi=sqrt(r*r-(x-cx)*(x-cx));
            xo=cy-xi;
            //factor 2*xi/(probhi[1]-problo[1]);
            if(i==0)
            height_arr(i,j,k,0)=xo;
            //      Print()<<"("<<i<<","<<0<<","<<k<<") y "<<height_arr(i,0,k,0)<<" "<<xi<<"xo"<<xo<<"cy"<<cx<<std::endl;
        });
        amrex::ParallelFor( tile_box011, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real y = problo[1]+(r[1]+j)*dx[1];
            Real cx  =problo[0]+0.5*(probhi[0]-problo[0]);
            Real cy  =problo[1]+0.5*(probhi[1]-problo[1]);
            Real yi, yo;
            Real r = (cx+cy)/2.0;
            if(y<cy)
                yi=sqrt(r*r-(cy-y)*(cy-y));
            else
                yi=sqrt(r*r-(y-cy)*(y-cy));
            yo=cx-yi;
            if(j==0)
            height_arr(i,j,k,1)=yo;
            //      Print()<<"("<<0<<","<<j<<","<<k<<") y "<<height_arr(0,j,k,1)<<" "<<yi<<"yo"<<yo<<"cx"<<cx<<std::endl;
        });
        amrex::ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = problo[0]+(r[0]+i)*dx[0];
            Real y = problo[1]+(r[1]+j)*dx[1];
            Real z = problo[2]+(r[2]+k)*dx[2];
            Real cx  =problo[0]+0.5*(probhi[0]-problo[0]);
            Real cy  =problo[1]+0.5*(probhi[1]-problo[1]);
            Real yi, yo;
            Real xi, xo;
            Real r = (cx+cy)/2.0;
            if(x<cx)
                xi=sqrt(r*r-(cx-x)*(cx-x));
            else
                xi=sqrt(r*r-(x-cx)*(x-cx));
            xo=cy-xi;

            if(y<cy)
                yi=sqrt(r*r-(cy-y)*(cy-y));
            else
                yi=sqrt(r*r-(y-cy)*(y-cy));
            yo=cx-yi;
            if(j!=0) {
                height_arr(i,j,k,0) = xo+2*xi/(probhi[1]-problo[1])*dx[1]*j;
            }
            if(i!=0) {
                height_arr(i,j,k,1) = yo+2*yi/(probhi[0]-problo[0])*dx[0]*i;
            }
            height_arr(i,j,k,2) = z;
            /*
            if(j==0) {
                Print()<<"("<<i<<","<<j<<","<<k<<") y "<<height_arr(i,j,k,1)<<" "<<yi<<std::endl;
            }
            if(i==0) {
                Print()<<"("<<i<<","<<j<<","<<k<<") x "<<height_arr(i,j,k,0)<<" "<<xi<<std::endl;
                }*/
            Real theta = x/(probhi[0]-problo[0])*(pi)*2.0-pi;
            Real radius = .25*cx+(y/(probhi[1]-problo[1]-.25*cx)*probhi[1])*0.25;
            height_arr(i,j,k,0)=cx+(radius*cos(theta));
            //use probhi[1] as the radius, swap it to something else if needed
            height_arr(i,j,k,1)=cy+(radius*sin(theta));
            /*
            if(theta>pi/2.0&&theta<pi) {
                height_arr(i,j,k,0)-=probhi[0];
                height_arr(i,j,k,1)-=probhi[1];
                height_arr(i,j,k,0)=0;
                height_arr(i,j,k,1)=0;
            } else if(theta>-pi/2.0&&theta<0) {
                height_arr(i,j,k,0)-=probhi[0];
                height_arr(i,j,k,0)=0;
                height_arr(i,j,k,1)=0;
            } else if(theta<=-pi/2.0&&false) {
                height_arr(i,j,k,1)-=probhi[1];
                height_arr(i,j,k,0)=0;
                height_arr(i,j,k,1)=0;
                }*/
        });
        /*
        amrex::ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real z = problo[2]+(r[2]+k)*dx[2];
            Real cx = problo[0]+0.5*(probhi[0]-problo[0]);
            Real cy = problo[1]+0.5*(probhi[1]-problo[1]);
            Real xi = cy-height_arr(i,0,k,0);
            Real yi = cx-height_arr(0,j,k,1);
            if(j<=1||j>=14) {
                height_arr(i,j,k,0) = height_arr(i,j-1,k,0)*NAN;
            }
            if(i<=1||i>=14) {
                height_arr(i,j,k,1) = height_arr(i-1,j,k,1)*NAN;
            }
            height_arr(i,j,k,2) = z;
        });
*/
        //      Print()<<FArrayBox(height_arr)<<std::endl;
    }
}
void
TerrainFittedPC::InitUmac (MultiFab* umac, int lev, Real dt, const MultiFab& a_z_height)
{
    BL_PROFILE("TerrainFittedPC::InitUmac");

    const Real* dx = Geom(lev).CellSize();
    const Real* plo = Geom(lev).ProbLo();
    auto domain = this->amrex::ParticleContainerBase::Geom(0).Domain();
    auto probhi = this->amrex::ParticleContainerBase::Geom(0).ProbHi();
    auto problo = this->amrex::ParticleContainerBase::Geom(0).ProbLo();
    Vector<std::unique_ptr<MultiFab> > raii_umac(AMREX_SPACEDIM);
    Vector<MultiFab*> umac_pointer(AMREX_SPACEDIM);
    if (OnSameGrids(lev, umac[0]))
    {
        for (int i = 0; i < AMREX_SPACEDIM; i++) {
            umac_pointer[i] = &umac[i];
        }
    }
    else
    {
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            int ng = umac[i].nGrow();
            raii_umac[i] = std::make_unique<MultiFab>
                (amrex::convert(m_gdb->ParticleBoxArray(lev), IntVect::TheDimensionVector(i)),
                 m_gdb->ParticleDistributionMap(lev), umac[i].nComp(), ng);
            umac_pointer[i] = raii_umac[i].get();
            umac_pointer[i]->ParallelCopy(umac[i],0,0,umac[i].nComp(),ng,ng);
        }
    }
    for(MFIter mfi(a_z_height); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.growntilebox();
        auto height_arr = a_z_height.array(mfi);
        auto umac_x_arr = umac[0].array(mfi);
        auto umac_y_arr = umac[1].array(mfi);
        auto umac_z_arr = umac[2].array(mfi);   
        //        Real r[3] = {0.5, 0.5, 0.5};  // this means place at cell center
        Real r[3] = {0.0, 0.0, 0.0};  // this means place at cell center
        const Real* dx = Geom(lev).CellSize();
        const Real* plo = Geom(lev).ProbLo();
        const Box tile_box101 = makeSlab(mfi.growntilebox(),1,0);
        const Box tile_box011 = makeSlab(mfi.growntilebox(),0,0);
        const Real pi=amrex::Math::pi<Real>();
        Real cx  =problo[0]+0.5*(probhi[0]-problo[0]);
        Real cy  =problo[1]+0.5*(probhi[1]-problo[1]);
        amrex::ParallelFor( tile_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = problo[0]+(r[0]+i)*dx[0];
            Real y = problo[1]+(r[1]+j)*dx[1];
            Real z = problo[2]+(r[2]+k)*dx[2];
            Real theta=atan((y-cy)/(x-cx));
            umac_x_arr(i,j,k,0)=cos(theta);//probhi[1];
            umac_y_arr(i,j,k,0)=sin(theta);//probhi[0];
            umac_z_arr(i,j,k,0)=0;
        });
    }
}

void
TerrainFittedPC::
InitParticles (MultiFab& a_z_height)
{
    BL_PROFILE("TerrainFittedPC::InitParticles");

    const int lev = 0;
    const Real* dx = Geom(lev).CellSize();
    const Real* plo = Geom(lev).ProbLo();
    auto domain = this->amrex::ParticleContainerBase::Geom(0).Domain();
    auto probhi = this->amrex::ParticleContainerBase::Geom(0).ProbHi();
    auto problo = this->amrex::ParticleContainerBase::Geom(0).ProbLo();

    for(MFIter mfi(a_z_height); mfi.isValid(); ++mfi)
    {
        const Box& tile_box  = mfi.tilebox();
        const auto& height = a_z_height[mfi];
        const FArrayBox* height_ptr = nullptr;
        auto height_arr = a_z_height.array(mfi);
#ifdef AMREX_USE_GPU
        std::unique_ptr<FArrayBox> hostfab;
        if (height.arena()->isManaged() || height.arena()->isDevice()) {
            hostfab = std::make_unique<FArrayBox>(height.box(), height.nComp(),
                                                  The_Pinned_Arena());
            Gpu::dtoh_memcpy_async(hostfab->dataPtr(), height.dataPtr(),
                                   height.size()*sizeof(Real));
            Gpu::streamSynchronize();
            height_ptr = hostfab.get();
        }
#else
        height_ptr = &height;
#endif
            Gpu::HostVector<ParticleType> host_particles;
            std::array<Gpu::HostVector<ParticleReal>, NAR> host_real;
            std::array<Gpu::HostVector<int>, NAI> host_int;
            const Real* dx = Geom(lev).CellSize();
            const Real* plo = Geom(lev).ProbLo();
            RealVect probhi(Geom(lev).ProbHi());
            RealVect problo(Geom(lev).ProbLo());
            //            std::vector<Gpu::HostVector<ParticleReal> > host_runtime_real(NumRuntimeRealComps());
            //            std::vector<Gpu::HostVector<int> > host_runtime_int(NumRuntimeIntComps());
        for (IntVect iv = tile_box.smallEnd(); iv <= tile_box.bigEnd(); tile_box.next(iv)) {
            if (iv[2] == 3) {
                Real r[3] = {0.0, 0.0, 0.0};  // this means place at cell center
                Real v[3] = {0.0, 0.0, 0.0};  // with 0 initial velocity

                Real x = (*height_ptr)(iv) + r[0]*((*height_ptr)(iv + IntVect(AMREX_D_DECL(1, 0, 0))) - (*height_ptr)(iv));
                Real y = (*height_ptr)(iv) + r[1]*((*height_ptr)(iv + IntVect(AMREX_D_DECL(0, 1, 0))) - (*height_ptr)(iv));
                Real z = (*height_ptr)(iv) + r[2]*((*height_ptr)(iv + IntVect(AMREX_D_DECL(0, 0, 1))) - (*height_ptr)(iv));
                int test=(probhi[0]-problo[0])/8.0;
                /*
                if(x-2*dx[0]<=problo[0]||y-2*dx[1]<=problo[1]||
                   x+2*dx[0]>=probhi[0]||y+2*dx[1]>=probhi[1])
                    continue;
                if((iv[0]-test)*dx[0]<=problo[0]||(iv[1]-test)*dx[1]<=problo[1]||
                   (iv[0]+test)*dx[0]>=probhi[0]||(iv[1]+test)*dx[1]>=probhi[1])
                    continue;
                */
                /*
                x = plo[0] + r[0] * height_arr(iv[0],iv[1],iv[2],0);
                y = plo[1] + r[1] * height_arr(iv[0],iv[1],iv[2],1);
                z = plo[2] + r[2] * height_arr(iv[0],iv[1],iv[2],2);

                x = height_arr(iv[0],iv[1],iv[2],0);
                y = height_arr(iv[0],iv[1],iv[2],1);
                z = height_arr(iv[0],iv[1],iv[2],2);
                */
                x = height_arr(iv[0],iv[1],iv[2],0)+r[0]*(height_arr(iv[0]+1,iv[1],iv[2],0)-height_arr(iv[0],iv[1],iv[2],0));
                y = height_arr(iv[0],iv[1],iv[2],1)+r[0]*(height_arr(iv[0],iv[1]+1,iv[2],1)-height_arr(iv[0],iv[1],iv[2],1));
                z = height_arr(iv[0],iv[1],iv[2],2)+r[0]*(height_arr(iv[0],iv[1],iv[2]+1,2)-height_arr(iv[0],iv[1],iv[2],2));

                ParticleType p;
                p.id()  = ParticleType::NextID();
                p.cpu() = ParallelDescriptor::MyProc();
                p.pos(0) = x;
                p.pos(1) = y;
                p.pos(2) = z;

                p.rdata(RealIdx::vx) = v[0];
                p.rdata(RealIdx::vy) = v[1];
                p.rdata(RealIdx::vz) = v[2];

                p.idata(IntIdx::i) = iv[0];  // particles carry their z-index
                p.idata(IntIdx::j) = iv[1];  // particles carry their z-index
                p.idata(IntIdx::k) = iv[2];  // particles carry their z-index
                //                amrex::Print()<<p<<" xyz "<<x<<" "<<y<<" "<<z<<" height "<<height_arr(iv[0],iv[1],iv[2],0)<<" "<<height_arr(iv[0],iv[1],iv[2],1)<<" "<<height_arr(iv[0],iv[1],iv[2],2)<<"prob "<<probhi<<"prob "<<problo<<std::endl;
                /*
                for (int i = NAR; i < NSR; ++i) p.rdata(i) = ParticleReal(p.id());
                for (int i = NAI; i < NSI; ++i) p.idata(i) = int(p.id());
                */
                host_particles.push_back(p);
                for (int i = 0; i < NAR; ++i)
                    host_real[i].push_back(p.rdata(i));
                for (int i = 0; i < NAI; ++i)
                    host_int[i].push_back(p.idata(i));
                /*
                for (int i = 0; i < NumRuntimeRealComps(); ++i)
                    host_runtime_real[i].push_back(p.rdata(NAR+i));
                for (int i = 0; i < NumRuntimeIntComps(); ++i)
                    host_runtime_int[i].push_back(p.idata(NAI+i)));
                */
           }
        }

            auto& particle_tile = DefineAndReturnParticleTile(lev, mfi.index(), mfi.LocalTileIndex());
            auto old_size = particle_tile.GetArrayOfStructs().size();
            auto new_size = old_size + host_particles.size();
            particle_tile.resize(new_size);

            Gpu::copyAsync(Gpu::hostToDevice,
                           host_particles.begin(),
                           host_particles.end(),
                           particle_tile.GetArrayOfStructs().begin() + old_size);

            auto& soa = particle_tile.GetStructOfArrays();
            for (int i = 0; i < NAR; ++i)
            {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_real[i].begin(),
                               host_real[i].end(),
                               soa.GetRealData(i).begin() + old_size);
            }

            for (int i = 0; i < NAI; ++i)
            {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_int[i].begin(),
                               host_int[i].end(),
                               soa.GetIntData(i).begin() + old_size);
            }
            /*
            for (int i = 0; i < NumRuntimeRealComps(); ++i)
            {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_runtime_real[i].begin(),
                               host_runtime_real[i].end(),
                               soa.GetRealData(NAR+i).begin() + old_size);
            }

            for (int i = 0; i < NumRuntimeIntComps(); ++i)
            {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_runtime_int[i].begin(),
                               host_runtime_int[i].end(),
                               soa.GetIntData(NAI+i).begin() + old_size);
            }
            */
            Gpu::streamSynchronize();
    }
    RedistributeLocal();
}

/*
  /brief Uses midpoint method to advance particles using umac.
*/
void
TerrainFittedPC::AdvectWithUmac (MultiFab* umac, int lev, Real dt, const MultiFab& a_z_height)
{
    BL_PROFILE("TerrainFittedPC::AdvectWithUmac()");
    AMREX_ASSERT(OK(lev, lev, umac[0].nGrow()-1));
    AMREX_ASSERT(lev >= 0 && lev < GetParticles().size());

    AMREX_D_TERM(AMREX_ASSERT(umac[0].nGrow() >= 1);,
                 AMREX_ASSERT(umac[1].nGrow() >= 1);,
                 AMREX_ASSERT(umac[2].nGrow() >= 1););

    AMREX_D_TERM(AMREX_ASSERT(!umac[0].contains_nan());,
                 AMREX_ASSERT(!umac[1].contains_nan());,
                 AMREX_ASSERT(!umac[2].contains_nan()););

    const auto      strttime = amrex::second();
    const Geometry& geom = m_gdb->Geom(lev);
    const Box& domain = geom.Domain();
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    Vector<std::unique_ptr<MultiFab> > raii_umac(AMREX_SPACEDIM);
    Vector<MultiFab*> umac_pointer(AMREX_SPACEDIM);
    if (OnSameGrids(lev, umac[0]))
    {
        for (int i = 0; i < AMREX_SPACEDIM; i++) {
            umac_pointer[i] = &umac[i];
        }
    }
    else
    {
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            int ng = umac[i].nGrow();
            raii_umac[i] = std::make_unique<MultiFab>
                (amrex::convert(m_gdb->ParticleBoxArray(lev), IntVect::TheDimensionVector(i)),
                 m_gdb->ParticleDistributionMap(lev), umac[i].nComp(), ng);
            umac_pointer[i] = raii_umac[i].get();
            umac_pointer[i]->ParallelCopy(umac[i],0,0,umac[i].nComp(),ng,ng);
        }
    }

    for (int ipass = 0; ipass < 2; ipass++)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (ParIterType pti(*this, lev); pti.isValid(); ++pti)
        {
            int grid    = pti.index();
            auto& ptile = ParticlesAt(lev, pti);
            auto& aos  = ptile.GetArrayOfStructs();
            const int n = aos.numParticles();
            auto *p_pbox = aos().data();
            const FArrayBox* fab[AMREX_SPACEDIM] = { AMREX_D_DECL(&((*umac_pointer[0])[grid]),
                                                                  &((*umac_pointer[1])[grid]),
                                                                  &((*umac_pointer[2])[grid])) };

            const auto& zheight_fab = a_z_height[grid];
            const auto zheight = zheight_fab.array();
            //array of these pointers to pass to the GPU
            amrex::GpuArray<amrex::Array4<const Real>, AMREX_SPACEDIM>
                const umacarr {{AMREX_D_DECL((*fab[0]).array(),
                                             (*fab[1]).array(),
                                             (*fab[2]).array() )}};

            amrex::ParallelFor(n,
                               [=] AMREX_GPU_DEVICE (int i)
            {
                ParticleType& p = p_pbox[i];
                if (p.id() <= 0) { return; }
                ParticleReal v[AMREX_SPACEDIM];
                mac_interpolate(p, plo, dxi, umacarr, v);
                if (ipass == 0)
                {
                    for (int dim=0; dim < AMREX_SPACEDIM; dim++)
                    {
                        p.rdata(dim) = p.pos(dim);
                        p.pos(dim) += static_cast<ParticleReal>(ParticleReal(0.5)*dt*v[dim]);
                    }
                }
                else
                {
                    for (int dim=0; dim < AMREX_SPACEDIM; dim++)
                    {
                        p.pos(dim) = p.rdata(dim) + static_cast<ParticleReal>(dt*v[dim]);
                        p.rdata(dim) = v[dim];
                    }

                    // also update z-coordinate here
                    IntVect iv(
                       AMREX_D_DECL(p.idata(0),
                                    p.idata(1),
                                    p.idata(2)));
                    auto xlo = zheight(iv[0], iv[1], iv[2],0);
                    auto xhi = zheight(iv[0]+1, iv[1], iv[2],0);
                    auto ylo = zheight(iv[0], iv[1], iv[2],1);
                    auto yhi = zheight(iv[0], iv[1]+1, iv[2],1);
                    auto zlo = zheight(iv[0], iv[1], iv[2],2);
                    auto zhi = zheight(iv[0], iv[1], iv[2]+1,2);
                    if (p.pos(0) > xhi) { // need to be careful here
                        p.idata(0) += 1;
                    } else if (p.pos(0) <= xlo) {
                        p.idata(0) -= 1;
                    }
                    if (p.pos(1) > yhi) { // need to be careful here
                        p.idata(1) += 1;
                    } else if (p.pos(1) <= ylo) {
                        p.idata(1) -= 1;
                    }
                    if (p.pos(2) > zhi) { // need to be careful here
                        p.idata(2) += 1;
                    } else if (p.pos(2) <= zlo) {
                        p.idata(2) -= 1;
                    }
                }
            });
        }
    }

    if (m_verbose > 1)
    {
        auto stoptime = amrex::second() - strttime;

#ifdef AMREX_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
                ParallelReduce::Max(stoptime, ParallelContext::IOProcessorNumberSub(),
                                    ParallelContext::CommunicatorSub());

                amrex::Print() << "TracerParticleContainer::AdvectWithUmac() time: " << stoptime << '\n';
#ifdef AMREX_LAZY
        });
#endif
    }
}
