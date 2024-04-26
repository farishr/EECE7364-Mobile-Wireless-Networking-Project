from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

# Function to simulate the Power Spectral Density (PSD) for different systems
def simulate_psd(frequencies, base_psd, lobe_width, lobe_height, noise_level):
    psd = np.full_like(frequencies, base_psd)
    main_lobe = np.abs(frequencies) <= lobe_width / 2
    psd[main_lobe] = lobe_height
    # Add random noise to simulate the variations in the PSD
    noise = np.random.normal(0, noise_level, size=frequencies.shape)
    psd += noise
    return psd

# Generate frequency data
num_points = 2048
frequencies = np.linspace(-0.5, 0.5, num_points, endpoint=False)

# Base PSD level and main lobe height for different systems
base_psd = -300
haar_lobe_height = -50
dmey_lobe_height = -60
db4_lobe_height = -70
sym4_lobe_height = -30
rbio_lobe_height = -60
coif1_lobe_height = -55
cf_ofdm_psd_level = -40

noise_level = 20

# Simulate PSD for Haar, dmey, and C-OFDM
haar_psd = simulate_psd(frequencies, base_psd, lobe_width=0.2, lobe_height=haar_lobe_height, noise_level=noise_level)
dmey_psd = simulate_psd(frequencies, base_psd, lobe_width=0.3, lobe_height=dmey_lobe_height, noise_level=noise_level)
db4_psd = simulate_psd(frequencies, base_psd, lobe_width=0.35, lobe_height=db4_lobe_height, noise_level=noise_level)
rbio_psd = simulate_psd(frequencies, base_psd, lobe_width=0.45, lobe_height=rbio_lobe_height, noise_level=noise_level)
db4_psd = simulate_psd(frequencies, base_psd, lobe_width=0.55, lobe_height=db4_lobe_height, noise_level=noise_level)
coif1_psd = simulate_psd(frequencies, base_psd, lobe_width=0.5, lobe_height=coif1_lobe_height, noise_level=noise_level)


cf_ofdm_psd = np.full_like(frequencies, cf_ofdm_psd_level)

# Plotting the PSDs
plt.figure(figsize=(20, 5))

# Plot for F-OWDM using different wavelets
plt.subplot(1, 3, 1)
plt.plot(frequencies, haar_psd, label='F-OWDM-haar')
plt.plot(frequencies, dmey_psd, label='F-OWDM-dmey')
plt.plot(frequencies, rbio_psd, label='F-OWDM-rbio')
plt.plot(frequencies, db4_psd, label ='F-OWDM-db4')
plt.plot(frequencies,coif1_psd, label='F-OWDM-coif4')
plt.plot(frequencies, cf_ofdm_psd, label='C-F-OFDM', linestyle='--')
plt.xlabel('Normalized frequency')
plt.ylabel('PSD (dB/Hz)')
plt.title('F-OWDM with Different Wavelets')
plt.legend()
plt.grid(True)

# Plot for comparison of OWDM with C-OFDM
plt.subplot(1, 3, 2)
plt.plot(frequencies, haar_psd, label='OWDM-haar')
plt.plot(frequencies, dmey_psd, label='OWDM-dmey')
plt.plot(frequencies, rbio_psd, label='F-OWDM-rbio')
plt.plot(frequencies, db4_psd, label ='F-OWDM-db4')
plt.plot(frequencies,coif1_psd, label='F-OWDM-coif4')
plt.plot(frequencies, cf_ofdm_psd, label='C-OFDM', linestyle='--')
plt.xlabel('Normalized frequency')
plt.title('OWDM and C-OFDM')
plt.legend()
plt.grid(True)

# Plot for comparison of F-OWDM with C-F-OFDM
plt.subplot(1, 3, 3)
plt.plot(frequencies, dmey_psd, label='F-OWDM-dmey')
plt.plot(frequencies, haar_psd, label='F-OWDM-haar')
plt.plot(frequencies, rbio_psd, label='F-OWDM-rbio')
plt.plot(frequencies, db4_psd, label ='F-OWDM-db4')
plt.plot(frequencies,coif1_psd, label='F-OWDM-coif4')
plt.plot(frequencies, cf_ofdm_psd, label='C-F-OFDM', linestyle='--')
plt.xlabel('Normalized frequency')
plt.title('F-OWDM and C-F-OFDM')
plt.legend()
plt.grid(True)

# Adjusting layout
plt.tight_layout()
plt.show()
