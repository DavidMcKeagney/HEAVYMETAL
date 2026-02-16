import JAC
using JAC
using JAC.Plasma   # plasma utilities

# Define plasma parameters
Te = 10.0  # electron temperature in eV
ne = 1e18  # electron number density in cm^-3

# Create a Plasma Environment object
plasma_env = Plasma.Environment(temperature = Te, density = ne)

# Choose an atomic target for which you want the plasma shift
# For example, hydrogen-like carbon
target_config = Atomic.Configuration("C5+") 

# Compute unperturbed energies
# Typical JAC code will first build the atomic structure:
levels = Atomic.levels(target_config)  # create atomic levels
energies_free = [lvl.energy for lvl in levels]

# Now compute plasma shifts for these levels
shifted_levels = Plasma.shift_levels(levels, plasma_env)

# Print results
for (lvl, shifted) in zip(levels, shifted_levels)
    println("Level: ", lvl, " Free energy: ", lvl.energy,
            " Plasma shifted: ", shifted.energy)
end



