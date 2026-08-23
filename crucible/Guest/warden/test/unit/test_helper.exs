# Ordinary ExUnit, for the parts of the control plane that are logic rather than
# timing: the edit matcher, path confinement, result ceilings. The simulation
# suite lives in test/dst and runs only in the :dst environment.
ExUnit.start()
