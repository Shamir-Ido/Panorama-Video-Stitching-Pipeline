# Space-Time Panorama Video
![panorama_middle_view](https://github.com/user-attachments/assets/f415c2ea-3b89-4092-bece-f5c1553ab611)

---

## Project Overview

This project explores **space-time panorama generation** from standard video footage. The goal is to synthesize new viewpoints by extracting vertical “slits” from consecutive frames and blending them into panoramas, creating a smooth transition over time.  

The core idea is to leverage **optical flow** to track camera motion, estimate translations between frames, and align the images globally. This allows me to reconstruct a coherent space-time view, where each panorama frame represents a different virtual perspective.

---

## Algorithm

The pipeline works in several stages:

### Automatic Motion Detection
- Analyzes a subset of initial frames to detect the main camera motion direction (e.g., left-to-right).  
- Computes the rotation angle needed to align all frames consistently with the detected motion.  

### Feature Tracking & Motion Estimation
- Tracks feature points using **Lucas–Kanade optical flow** between consecutive frames.  
- Computes median translation vectors `(tx, ty)` for robust camera motion estimation.  
- Accumulates translations across frames to build a global transformation per frame.  

### Global Alignment & Drift Correction
- Applies accumulated transformations to position frames on a common canvas.  
- Corrects vertical drift to maintain a stable horizon throughout the panorama.  

### Panorama Slicing & Blending
- Extracts vertical strips from aligned frames at different column offsets.  
- Blends overlapping strips using **feathering** to minimize visible seams.  

### Final Panorama Generation
- Generates a sequence of panoramas representing multiple virtual viewpoints.  
- Slight Gaussian smoothing is applied to reduce residual artifacts and improve visual coherence.

---

## Implementation Highlights

- Built as a **step-by-step pipeline**, processing input frames from motion detection to final panorama generation.  
- **Lucas–Kanade optical flow** was used for motion estimation and feature tracking.  
- Median translation values mitigate the effect of outliers in motion estimation.  
- Global drift correction ensures the panoramas remain horizontally stable.  
- Slicing and **feathered blending** reduce seams and produce smooth viewpoint transitions.  
- Automatic motion direction detection allows the algorithm to adapt to different camera orientations.  

Libraries used:
- **OpenCV:** feature detection, optical flow, image rotation, Gaussian blur  
- **NumPy:** matrix operations, translation accumulation  
- **PIL:** final image formatting  

Empirical parameters:
- Tracked features: 500–1500  
- Feathering width: small overlap to smooth seams  
- Slice positions: 10%–90% of image width  
- Gaussian blur: small kernel for final smoothing  

---

## Results

### Successful Video Examples
- Generated panoramas capture **different virtual viewpoints** with smooth transitions.  
- Main structures are clearly aligned, and the motion is consistent across frames.  
- The method effectively tracks camera translation and synthesizes new perspectives.

### Challenges & Limitations
- **Parallax artifacts:** Ghosting occurs on objects at different depths.  
- **Temporal aliasing:** Fast camera motion can cause slight vertical bands.  
- **Zoom or non-linear motion:** Input videos with zoom or strong depth variation produce distorted panoramas because the linear translation assumption is broken.  
- Minor edge distortions and blending artifacts remain due to optical flow noise and lighting variations.  

### Observations
- The method works best with videos that have **linear camera translation** and limited depth variation.  
- Videos with zoom or strong parallax break the core assumptions, leading to noticeable distortions.  

---

## Conclusion

This project demonstrates how **space-time panorama videos** can be generated from ordinary video footage using optical flow and global motion accumulation.  

- Successfully creates smooth changing-viewpoint panoramas.  
- Highlights both the **potential** of optical-flow-based synthesis and its **limitations** in complex motion scenarios.  
- Offers insights into motion estimation, image alignment, and seamless blending techniques in computer vision applications.
