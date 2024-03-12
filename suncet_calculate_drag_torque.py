import numpy as np
import astropy.units as u
from astropy.constants import G, M_earth, R_earth
import argparse
from suncet_get_spatial_average_atmospheric_density import average_atmospheric_density

def calculate_orbital_velocity(altitude):
    altitude_m = altitude.to(u.m)
    r = R_earth + altitude_m  # Distance from the center of the Earth
    v = np.sqrt(G * M_earth / r)
    return v

def main(altitude_km, solar_conditions):
    altitude = altitude_km * u.km

    # Constants
    coefficient_of_drag = 2.5 * u.dimensionless_unscaled
    asymmetric_surface_area = 0.144 * u.m**2
    torque_lever_arm = (0.36 + (0.1 - 0.061)) * u.m
    torque_rod_mag_field_angle = 90 * u.deg

    # XACT-15 specs
    adcs_momentum_storage = 0.015 * u.N * u.m * u.s
    torque_rod_dipole_moment = 0.2 * u.A * u.m**2
    earth_mag_field_strength_in_leo = 50e-6 * u.T
    torque_rod_torque = (torque_rod_dipole_moment * earth_mag_field_strength_in_leo * np.sin(torque_rod_mag_field_angle.to(u.radian))).to(u.N * u.m)

    # Calculated quantities
    atmospheric_density = average_atmospheric_density(altitude.value, solar_conditions) * u.kg / u.m**3
    velocity = calculate_orbital_velocity(altitude)
    torque_drag = (1/2 * atmospheric_density * velocity**2 * coefficient_of_drag * asymmetric_surface_area * torque_lever_arm).to(u.N * u.m)
    time_to_saturate_wheels = (adcs_momentum_storage / torque_drag).to(u.minute)

    if torque_drag < torque_rod_torque:
        print('The disturbance torque is less than what any individual torque rod can dump.')
    else:
        print(f'The disturbance torque is greater than what the torque rods can dump. The wheels will saturate in {time_to_saturate_wheels:.2f} (assuming best case that they started from rest).')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate satellite torque and wheel saturation time.')
    parser.add_argument('altitude', type=float, help='Altitude in kilometers')
    parser.add_argument('solar_conditions', choices=['min', 'mean', 'max'], help='Solar conditions: min, mean, or max')

    args = parser.parse_args()

    main(args.altitude, args.solar_conditions)
