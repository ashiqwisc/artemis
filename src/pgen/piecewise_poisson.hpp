#ifndef PGEN_PIECEWISE_POISSON_HPP_
#define PGEN_PIECEWISE_POISSON_HPP_
//! \file piecewise_poisson.hpp

// C/C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"
#include "utils/artemis_utils.hpp"

namespace piecewise_poisson { // much of this is ported from polytrope.hpp
//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::LinearWave_()
//! \brief Sets density and subdomains for piecewise constant Poisson eq. tests
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
    using parthenon::MakePackDescriptor;
    const Mesh *pmesh = pmb->pmy_mesh;
    const int ndim = pmesh->ndim;
    using TE = parthenon::TopologicalElement;

    // packing and capture variables for kernel
    auto &md = pmb->meshblock_data.Get();
    for (auto &var : md->GetVariableVector()) {
    if (!var->IsAllocated()) pmb->AllocateSparse(var->label());
    }
    static auto desc =
        MakePackDescriptor<gas::prim::density, gas::prim::velocity, gas::prim::sie>(
            (pmb->resolved_packages).get());
    auto v = desc.GetPack(md.get());
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    auto &pco = pmb->coords;
    const auto &cpars =
        pmb->packages.Get("artemis")->template Param<geometry::CoordParams>("coord_params");

    // Configuration checks 
    auto &artemis_pkg = pmb->packages.Get("artemis");
    PARTHENON_REQUIRE(artemis_pkg->Param<std::bool>("do_gas"),
                    "Enable gas package");
    auto &gas_pkg = pmb->packages.Get("gas");
    PARTHENON_REQUIRE(artemis_pkg->Param<bool>("do_self_gravity"),
                    "Enable self_gravity package");
    auto &grav_pkg = pmb->packages.Get("self_gravity");
    // Check 4piG=1 (from grav_slab.hpp)
    PARTHENON_REQUIRE(grav_pkg->Param<Real>("four_pi_G") == 1.0,
                    "piecewise_poisson requires 4piG=1 via self_gravity/units_override=true");
    // Check Cartesian geometry 
    PARTHENON_REQUIRE(GEOM == Coordinates::cartesian,
                "Need cartesian coordinates");
    // Check gas EOS is ideal gas 
    PARTHENON_REQUIRE(gas_pkg->Param<std::string>("eos_type") == "ideal",
                    "EOS needs to be ideal gas EOS");
    const Real adiabatic_index = gas_pkg->Param<std::const>("adiabatic_index");

    // TODO: Set piecewise constant problem fields in MeshBlock (which is the object describing the mesh btw). Make sure to init density properly 

    // Polytrope params
    const int iprob = pin->GetOrAddInteger("problem", "iprob", 1);
    PARTHENON_REQUIRE((iprob == 1) || (iprob == 2), "iprob not recognized!");
    const Real x10a = pin->GetOrAddReal("problem", "x10a", (iprob == 2) * 4.0);
    const Real x20a = pin->GetOrAddReal("problem", "x20a", (iprob == 2) * 2.5);
    const Real x30a = pin->GetOrAddReal("problem", "x30a", (iprob == 2) * 0.0);
    const Real x10b = pin->GetOrAddReal("problem", "x10b", -4.0);
    const Real x20b = pin->GetOrAddReal("problem", "x20b", -2.5);
    const Real x30b = pin->GetOrAddReal("problem", "x30b", 0.0);
    const Real rho_amb = pin->GetOrAddReal("problem", "rho_amb", 1.0e-3);
    const Real sie_amb = pin->GetOrAddReal("problem", "sie_amb", 1.0e2);

    // NOTE(@pdmullen): Hardcoded params related to n=1 polytrope profile...
    const Real alpha = std::sqrt(0.5);
    const Real cutoff = 0.75 * M_PI;

    // Polytrope init
    pmb->par_for(
        "polytrope", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // cell-centered coordinates
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto &xv = coords.GetCellCenter();

        // Compute Lane-Emden profiles
        const Real ar1 =
            alpha * std::sqrt(SQR(xv[0] - x10a) + SQR(xv[1] - x20a) + SQR(xv[2] - x30a));
        const Real ar2 =
            alpha * std::sqrt(SQR(xv[0] - x10b) + SQR(xv[1] - x20b) + SQR(xv[2] - x30b));
        const Real lane_emden1 = std::sin(ar1) / ar1;
        const Real lane_emden2 = std::sin(ar2) / ar2;
        const bool inside1 = (ar1 < cutoff);
        const bool inside2 = (ar2 < cutoff);

        // Initialize polytrope(s)
        Real trho = Null<Real>(), tsie = Null<Real>();
        if (iprob == 1) {
            trho = inside1 ? lane_emden1 : rho_amb;
            tsie = inside1 ? lane_emden1 : sie_amb;
        } else if (iprob == 2) {
            trho = inside1 ? lane_emden1 : (inside2 ? lane_emden2 : rho_amb);
            tsie = inside1 ? lane_emden1 : (inside2 ? lane_emden2 : sie_amb);
        }
        v(0, gas::prim::density(), k, j, i) = trho;
        v(0, gas::prim::sie(), k, j, i) = tsie;
        v(0, gas::prim::velocity(0), k, j, i) = (iprob == 2) * (inside2 - inside1);
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        });
}

} // namespace piecewise_poisson

#endif // PGEN_PIECEWISE_POISSON_HPP_
