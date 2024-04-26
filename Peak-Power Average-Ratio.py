import numpy as np
import pywt
import matplotlib.pyplot as plt

def generate_ofdm_signal(n_subcarriers, wavelet, use_wavelet):
    # Generate random QPSK symbols
    symbols = (np.random.randint(0, 2, n_subcarriers) * 2 - 1) + 1j * (np.random.randint(0, 2, n_subcarriers) * 2 - 1)
    
    if use_wavelet:
        # Apply Inverse Discrete Wavelet Transform
        coeffs = pywt.wavedec(symbols, wavelet)
        ofdm_signal = pywt.waverec(coeffs, wavelet)
        print(f"generated Output with the implementation of IDWT")
    else:
        # Apply IFFT for traditional OFDM
        ofdm_signal = np.fft.ifft(symbols)
    
    return ofdm_signal

def calculate_papr(signal):
    peak_power = np.max(np.abs(signal)**2)
    avg_power = np.mean(np.abs(signal)**2)
    papr = 10 * np.log10(peak_power / avg_power)
    return papr

# Simulation parameters
n_subcarriers = 256
eb_no_dbs = np.linspace(1, 10, 10)  # Eb/No range in dB
wavelets = ['haar', 'db4', 'sym4', 'rbio2.2','coif1', 'none']  # 'none' for traditional OFDM
papr_results = {wavelet: [] for wavelet in wavelets}

# Perform simulations
for wavelet in wavelets:
    for eb_no_db in eb_no_dbs:
        ofdm_signal = generate_ofdm_signal(n_subcarriers, wavelet, use_wavelet=(wavelet != 'none'))
        papr = calculate_papr(ofdm_signal)
        papr_results[wavelet].append(papr)

# Plot PAPR results
plt.figure(figsize=(10, 6))
for wavelet, papr_values in papr_results.items():
    if wavelet == 'none':
        plt.plot(eb_no_dbs, papr_values, 'o--', label='OFDM')
    else:
        plt.plot(eb_no_dbs, papr_values, 'x-', label=wavelet)
    
plt.yscale('log')
plt.xlabel('EB/No (dB)')
plt.ylabel('Peak-to-average power ratio')
plt.title('PAPR for different wavelets in Wavelet OFDM')
plt.legend()
plt.grid(True)
plt.show()