import numpy as np 
import astropy.units as u
from astropy.constants import G, M_earth, R_earth
from suncet_get_spatial_average_atmospheric_density import average_atmospheric_density


# User input (main things that will be exposed by eventual function)
altitude = 440 *u.km
solar_conditions = 'max'

# Constants -- really these are tuneable inputs too, but don't expect to change them much for SunCET
coefficient_of_drag = 2.5 * u.dimensionless_unscaled # depends on shape but 2.5 is typical
asymmetric_surface_area = 0.144 * u.m**2 # area normal to velocity vector, e.g., the solar array sailing against the wind imparting torque in one direction about the spacecraft
torque_lever_arm = (0.36 + (0.1 - 0.061)) * u.m # distance from centroid of the asymmetric protrusion (e.g., the solar arrays) to center of mass. 0.36 m [one panel out to get to array center] + (0.1 - 0.061) m [from +Y wall to deployed cg based on CDR Bus slide 13]
torque_rod_mag_field_angle = 90 * u.deg # angle between the torque rod and the Earth's magnetic field at the location of the spacecraft. Best case scenario is 90º which will provide max torque. 

# XACT-15 specs
adcs_momentum_storage = 0.015 * u.N *u.m * u.s # amount of angular momentum the system can store. The spec sheet says that this is for the whole system, though it's not clear if each wheel can hold this much or if each is less and it knows how to transition the momentum between them. 
torque_rod_dipole_moment = 0.2 * u.A *u.m**2 # basically the "strength" of each torque rod
earth_mag_field_strength_in_leo = 50e-6 * u.T # it actually varies but this is a typical value
torque_rod_torque = (torque_rod_dipole_moment * earth_mag_field_strength_in_leo * np.sin(torque_rod_mag_field_angle.to(u.radian))).to(u.N * u.m)


def calculate_orbital_velocity(altitude):
    altitude_m = altitude.to(u.m)
    r = R_earth + altitude_m # Distance from center of the Earth

    v = np.sqrt(G * M_earth / r)
    return v


# Calcualted intermediate quantities
atmospheric_density = average_atmospheric_density(altitude.value, solar_conditions) * u.kg / u.m**3
# Example density: 3.23e-11 * u.kg/u.m**3 # user input for now
velocity = calculate_orbital_velocity(altitude)

# Calculate main values of interest
torque_drag = (1/2 * atmospheric_density * velocity**2 * coefficient_of_drag * asymmetric_surface_area * torque_lever_arm).to(u.N * u.m)
time_to_saturate_wheels = (adcs_momentum_storage/torque_drag).to(u.minute) # From wheel at 0 speed, how long until it saturates with disturbance torques above?

if torque_drag < torque_rod_torque: 
    print('The disturbance torque is less than what any individual torque rod can dump')
else: 
    print('The disturbance torque is gretter than what the torque rods can dump. The wheels will saturate in {:.2f} (assuming best case that they started from rest).'.format(time_to_saturate_wheels.to(u.minute)))
