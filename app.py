import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # ✅ Force an interactive GUI backend
import matplotlib.pyplot as plt
from spectral import open_image
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class PCAViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DataLens Free PCA Viewer")
        self.root.geometry("400x300")

        self.label = tk.Label(root, text="Upload ENVI .hdr File:", font=('Helvetica', 11, 'bold'))
        self.label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="Browse", command=self.load_hdr)
        self.upload_btn.pack(pady=5)

        self.variance_btn = tk.Button(root, text="Show Top 100 Variance Map", command=self.show_top_variance)
        self.variance_btn.pack(pady=10)

        self.spectrum_btn = tk.Button(root, text="Show Average Spectrum", command=self.show_average_spectrum)
        self.spectrum_btn.pack(pady=10)

        self.pca_result = None
        self.flattened_data = None
        self.cropped_wavelengths = None
        self.top_variance_pixels = None
        self.height = None
        self.width = None

    def load_hdr(self):
        file_path = filedialog.askopenfilename(filetypes=[("HDR files", "*.hdr")])
        if not file_path:
            return

        try:
            img = open_image(file_path)
            datacube = img.load()

            wavelengths = np.array(img.metadata['wavelength']).astype(float)
            crop_indices = np.where((wavelengths >= 715) & (wavelengths <= 1700))[0]

            cropped_datacube = datacube[:, :, crop_indices]
            self.cropped_wavelengths = wavelengths[crop_indices]

            self.height, self.width, bands = cropped_datacube.shape
            self.flattened_data = cropped_datacube.reshape(-1, bands)

            # Normalize
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(self.flattened_data)

            # PCA
            pca = PCA(n_components=5)
            self.pca_result = pca.fit_transform(normalized_data)
            pca_image = self.pca_result[:, 0].reshape(self.height, self.width)

            # Show PCA image
            plt.figure(figsize=(6, 5))
            plt.imshow(pca_image, cmap="viridis")
            plt.title("PCA - First Component")
            plt.colorbar()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load and process HDR file.\n\n{e}")

    def show_top_variance(self):
        if self.pca_result is None or self.height is None or self.width is None:
            messagebox.showwarning("No PCA", "Please upload a file and perform PCA first.")
            return

        total_pixels = self.pca_result.shape[0]
        self.top_variance_pixels = np.argsort(self.pca_result[:, 0])[-100:]

        # Ensure indices are valid
        self.top_variance_pixels = self.top_variance_pixels[self.top_variance_pixels < total_pixels]

        highlight_map = np.zeros((total_pixels,))
        highlight_map[self.top_variance_pixels] = 1
        highlight_map = highlight_map.reshape(self.height, self.width)

        plt.figure(figsize=(6, 5))
        plt.imshow(highlight_map, cmap="cool", alpha=0.8)
        plt.title("Highlighted Map of Top Variance Locations")
        plt.xlabel("Sample Index (Width)")
        plt.ylabel("Line Index (Height)")
        plt.show()

    def show_average_spectrum(self):
        if self.top_variance_pixels is None:
            messagebox.showwarning("No Highlighted Points", "Please compute and display variance map first.")
            return

        highlighted_spectra = self.flattened_data[self.top_variance_pixels, :]
        average_spectrum = np.mean(highlighted_spectra, axis=0)
        normalized_avg = (average_spectrum - np.min(average_spectrum)) / (np.max(average_spectrum) - np.min(average_spectrum))

        plt.figure(figsize=(7, 5))
        plt.plot(self.cropped_wavelengths, normalized_avg, label="Normalized Average Spectrum", color="blue")
        plt.title("Normalized Average Spectrum of Highlighted Components")
        plt.xlabel("Wavenumbers (cm⁻¹)")
        plt.ylabel("Normalized Intensity (0–1)")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = PCAViewerApp(root)
    root.mainloop()
