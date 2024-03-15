import numpy as np 
import astropy.units as u
from astropy.constants import G, M_earth, R_earth
import pandas as pd
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from suncet_get_spatial_average_atmospheric_density import average_atmospheric_density


# User input (main things that will be exposed by eventual function)
altitude = 400 * u.km
solar_conditions = 'max'

# Constants -- really these are tuneable inputs too, but don't expect to change them much for SunCET
coefficient_of_drag = 2.5 * u.dimensionless_unscaled # depends on shape but 2.5 is typical
asymmetric_surface_area = 0.144 * u.m**2 # area normal to velocity vector, e.g., the solar array sailing against the wind imparting torque in one direction about the spacecraft
#torque_lever_arm = (0.36 + (0.1 - 0.061)) * u.m # distance from centroid of the asymmetric protrusion (e.g., the solar arrays) to center of mass. 0.36 m [one panel out to get to array center] + (0.1 - 0.061) m [from +Y wall to deployed cg based on CDR Bus slide 13]
torque_lever_arm = 0.203 * u.m # distance from centroid of the asymmetric protrusion (e.g., the solar arrays) to center of mass. Evan's idea to dual deploy by first panel normal, second panel unfolds in perpindicular direction
torque_rod_mag_field_angle = 90 * u.deg # angle between the torque rod and the Earth's magnetic field at the location of the spacecraft. Best case scenario is 90º which will provide max torque. 

# Input data
orbit_sun_vector_to_velocity_vector_angle = pd.read_csv('/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/data/synthetic/stk_orbit/SunCET_Sun_Direction_to_Velocity_Direction_400km_Noon_Midnight.csv')

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
velocity = calculate_orbital_velocity(altitude)
asymmetric_surface_area_projected_vs_time = asymmetric_surface_area * np.cos(np.radians(orbit_sun_vector_to_velocity_vector_angle['Angle (deg)'].values))
orbit_sun_vector_to_velocity_vector_angle['Time (UTCG)'] = pd.to_datetime(orbit_sun_vector_to_velocity_vector_angle['Time (UTCG)'], format="%d %b %Y %H:%M:%S.%f")
minutes_since_start = ((orbit_sun_vector_to_velocity_vector_angle['Time (UTCG)'] - orbit_sun_vector_to_velocity_vector_angle['Time (UTCG)'].iloc[0]).dt.total_seconds() / 60).values

# Calculate main values of interest
torque_drag_peak = (1/2 * atmospheric_density * velocity**2 * coefficient_of_drag * asymmetric_surface_area * torque_lever_arm).to(u.N * u.m)
time_to_saturate_wheels_if_always_peak_drag = (adcs_momentum_storage/torque_drag_peak).to(u.minute) # From wheel at 0 speed, how long until it saturates with disturbance torques above?
torque_drag_vs_time = (1/2 * atmospheric_density * velocity**2 * coefficient_of_drag * asymmetric_surface_area_projected_vs_time * torque_lever_arm).to(u.N * u.m)
cumulative_torque_drag = cumtrapz(torque_drag_vs_time, (minutes_since_start * 60), initial=0) * torque_drag_vs_time.unit * u.s

cumulative_torque_rod = np.zeros_like(cumulative_torque_drag)
net_momentum = np.zeros_like(cumulative_torque_drag)
current_cumulative_torque_rod = 0
for i in range(1, len(minutes_since_start)):
    if net_momentum[i-1] < 0: 
        direction_of_torque_rod = 1
    else: 
        direction_of_torque_rod = -1
    cumulative_torque_rod[i] = direction_of_torque_rod * torque_rod_torque * ((minutes_since_start[i] - minutes_since_start[i-1]) * u.minute).to(u.s)
    cumulative_torque_rod[i] += cumulative_torque_rod[i-1] # since drag is cumulative, also need rods to be cumulative in order to compute the net
    net_momentum[i] = cumulative_torque_drag[i] + cumulative_torque_rod[i]

target_index = np.argmax(net_momentum >= adcs_momentum_storage)
target_time = minutes_since_start[target_index] if net_momentum[target_index] >= adcs_momentum_storage else None


# Plot the torques
fig, axs = plt.subplots(4, 1, figsize=(8, 13))
axs[0].plot(minutes_since_start, torque_drag_vs_time)
axs[0].set_title('SunCET {} km, noon-midnight, {}'.format(altitude.value, solar_conditions))
axs[0].set_ylabel('drag torque [Nm]')
axs[0].grid(True)

axs[1].plot(minutes_since_start, cumulative_torque_drag)
axs[1].set_ylabel('cumulative drag momentum [Nms]')
axs[1].grid(True)

axs[2].plot(minutes_since_start, cumulative_torque_rod)
axs[2].set_ylabel('cumulative torque rod momentum [Nms]')
axs[2].grid(True)

axs[3].plot(minutes_since_start, net_momentum)
axs[3].set_xlabel('time [minutes since start]')
axs[3].set_ylabel('net momentum [Nms]')
axs[3].grid(True)

axs[3].axhline(y=adcs_momentum_storage.value, color='r', linestyle='--', label=f'XACT-15 angular momentum storage: {adcs_momentum_storage.value} Nms')
axs[3].axhline(y=-adcs_momentum_storage.value, color='r', linestyle='--')
axs[3].set_ylim([-0.020, 0.020])
axs[3].legend()

plt.tight_layout()

if torque_drag_peak < torque_rod_torque: 
    print('The disturbance torque is less than what any individual torque rod can dump')
else: 
    print('The disturbance torque is greater than what the torque rods can dump. The wheels will saturate in {:.2f} minutes (assuming best case that they started from rest).'.format(target_time))


pass