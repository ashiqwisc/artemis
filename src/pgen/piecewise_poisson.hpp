#ifndef PGEN_PIECEWISE_POISSON_HPP_
#define PGEN_PIECEWISE_POISSON_HPP_
//! \file piecewise_poisson.hpp
//! \brief Initializes a two-region piecewise-constant density for a Poisson solve.

// C++ headers
#include <string>

// Artemis headers
#include "artemis.hpp"
#include "geometry/geometry.hpp"

namespace piecewise_poisson {
//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin)
//! \brief Sets a two-region piecewise-constant density and auxiliary gas state.
template <Coordinates GEOM>
inline void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  PARTHENON_INSTRUMENT
  using parthenon::MakePackDescriptor;

  // Configuration checks
  PARTHENON_REQUIRE(GEOM == Coordinates::cartesian,
                    "piecewise_poisson requires Cartesian coordinates");

  auto &artemis_pkg = pmb->packages.Get("artemis");
  PARTHENON_REQUIRE(artemis_pkg->Param<bool>("do_gas"),
                    "piecewise_poisson requires the gas package");
  PARTHENON_REQUIRE(artemis_pkg->Param<bool>("do_self_gravity"),
                    "piecewise_poisson requires the self_gravity package");

  auto &gas_pkg = pmb->packages.Get("gas");
  auto &grav_pkg = pmb->packages.Get("self_gravity");
  PARTHENON_REQUIRE(gas_pkg->Param<int>("nspecies") == 1,
                    "piecewise_poisson requires one gas species");
  PARTHENON_REQUIRE(gas_pkg->Param<std::string>("eos_type") == "ideal",
                    "piecewise_poisson requires an ideal-gas EOS");
  PARTHENON_REQUIRE(grav_pkg->Param<Real>("four_pi_G") == 1.0,
                    "piecewise_poisson requires 4piG=1");

  const Real gamma = gas_pkg->Param<Real>("adiabatic_index");
  PARTHENON_REQUIRE(gamma > 1.0,
                    "piecewise_poisson requires an adiabatic index greater than one");
  const Real gm1 = gamma - 1.0;

  // Piecewise-constant density and auxiliary pressure parameters
  const int nsubdomains = pin->GetInteger("problem", "subdomains");
  const Real rho1 = pin->GetReal("problem", "rho_1");
  const Real rho2 = pin->GetReal("problem", "rho_2");
  const Real threshold = pin->GetReal("problem", "threshold");
  const Real pressure = pin->GetReal("problem", "auxiliary_pressure");
  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");

  PARTHENON_REQUIRE(nsubdomains == 2,
                    "piecewise_poisson currently supports exactly two subdomains");
  PARTHENON_REQUIRE(rho1 > 0.0 && rho2 > 0.0,
                    "piecewise_poisson requires positive densities");
  PARTHENON_REQUIRE(pressure > 0.0,
                    "piecewise_poisson requires positive auxiliary pressure");
  PARTHENON_REQUIRE(x1min < threshold && threshold < x1max,
                    "piecewise_poisson threshold must lie strictly inside the x1 domain");

  // Pack and capture variables for the initialization kernel
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
      artemis_pkg->template Param<geometry::CoordParams>("coord_params");

  // Initialize rho=rho1 for x1<threshold and rho=rho2 otherwise. 
  pmb->par_for(
      "pgen_piecewise_poisson", kb.s, kb.e, jb.s, jb`.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        geometry::Coords<GEOM> coords(cpars, pco, k, j, i);
        const auto &xv = coords.GetCellCenter();
        const Real rho = (xv[0] < threshold) ? rho1 : rho2;
        const Real sie = pressure / (gm1 * rho); // ideal gas law, not really used in our dataset but necessary or this repo 

        v(0, gas::prim::density(), k, j, i) = rho;
        v(0, gas::prim::velocity(0), k, j, i) = 0.0;
        v(0, gas::prim::velocity(1), k, j, i) = 0.0;
        v(0, gas::prim::velocity(2), k, j, i) = 0.0;
        v(0, gas::prim::sie(), k, j, i) = sie;
      });
}

} // namespace piecewise_poisson

#endif // PGEN_PIECEWISE_POISSON_HPP_
