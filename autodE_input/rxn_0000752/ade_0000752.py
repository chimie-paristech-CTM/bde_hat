import autode as ade
import time
t0 = time.time()
ade.Config.n_cores=24
ade.Config.max_core=4000
ade.Config.hcode="G16"
ade.Config.lcode ="xtb"
rxn=ade.Reaction(r"CC(C)CCNCCO.[O][C@@H]1C=CCCO1>>CC(C)CCNCC[O].O[C@@H]1C=CCCO1")
ade.Config.G16.keywords.set_functional('um062x')
ade.Config.G16.keywords.set_dispersion(None)
kwds_sp = ade.Config.G16.keywords.sp
kwds_sp.append(' stable=opt')
ade.Config.num_conformers=1000
ade.Config.rmsd_threshold=0.1
ade.Config.hmethod_conformers=True
rxn.calculate_reaction_profile(free_energy=True)
t1 = time.time()
print(f"Duration: {t1-t0}")
