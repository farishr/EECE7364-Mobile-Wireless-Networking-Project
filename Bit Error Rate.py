# Adjusted code for plotting BER vs SNR for different wavelet types in a single graph
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Parameters
n_subcarriers = 64
n_symbols = 100  # Reduce the number of symbols to avoid execution issues
wavelet_types = ['haar', 'db4', 'sym4','rbio2.2','coif1']
bits_per_symbol = 2 #QPSK
SNR_dB_range = np.arange(0, 21, 5)  # Reduced SNR range for demonstration

# Function to generate QPSK symbols
def generate_qpsk_symbols(n_symbols, n_subcarriers):
    bits = np.random.randint(0, 2, (n_symbols, n_subcarriers, bits_per_symbol))
    symbols = (2*bits[:, :, 0] - 1) + 1j * (2*bits[:, :, 1] - 1)
    return bits, symbols.flatten()

# Function to calculate BER
def calculate_ber(bits, rx_bits):
    bit_errors = np.sum(bits != rx_bits)
    total_bits = np.product(bits.shape)
    return bit_errors / total_bits

# BER simulation for different wavelets
ber_results = {wavelet: [] for wavelet in wavelet_types}

# Simulation loop
for wavelet in wavelet_types:
    for SNR_dB in SNR_dB_range:
        # Generate data
        bits, data = generate_qpsk_symbols(n_symbols, n_subcarriers)
        
        # Wavelet OFDM Modulation
        tx_signal = np.zeros_like(data, dtype=complex)
        for i in range(0, len(data), n_subcarriers):
            coeffs = pywt.wavedec(data[i:i+n_subcarriers], wavelet, level=int(np.log2(n_subcarriers)))
            tx_signal[i:i+n_subcarriers] = pywt.waverec(coeffs, wavelet)
        
        # Channel
        SNR_linear = 10**(SNR_dB/10.0)
        noise_variance = n_subcarriers / (2 * SNR_linear)
        noise = np.sqrt(noise_variance) * (np.random.randn(len(data)) + 1j * np.random.randn(len(data)))
        rx_signal = tx_signal + noise
        
        # Wavelet OFDM Demodulation
        rx_data = np.zeros_like(data, dtype=complex)
        for i in range(0, len(data), n_subcarriers):
            rx_sample = rx_signal[i:i+n_subcarriers]
            coeffs_from_rx = pywt.wavedec(rx_sample, wavelet, level=int(np.log2(n_subcarriers)))
            rx_data[i:i+n_subcarriers] = pywt.waverec(coeffs_from_rx, wavelet)
        
        # Symbol to bit mapping
        rx_bits = np.array([rx_data.real > 0, rx_data.imag > 0]).astype(int).T
        rx_bits = rx_bits.reshape(-1, n_subcarriers, bits_per_symbol)
        
        # BER calculation
        ber = calculate_ber(bits, rx_bits)
        ber_results[wavelet].append(ber)

# Plotting
plt.figure(figsize=(12, 8))
for wavelet in wavelet_types:
    plt.semilogy(SNR_dB_range, ber_results[wavelet], marker='o', label=wavelet)

plt.title('BER vs SNR for Various Wavelet Types in Wavelet OFDM')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.legend()
plt.grid(True)
plt.show()
