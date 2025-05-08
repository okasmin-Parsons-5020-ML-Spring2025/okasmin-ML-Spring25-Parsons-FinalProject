const container = document.getElementById("image-container");

const loadImagesClusterList = async () => {
  try {
    const res = await fetch("data/sortedImages.json");
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

    const filenames = await res.json();
    container.innerHTML = ""; // clear if needed

    filenames.forEach((filename) => {
      const img = document.createElement("img");
      img.src = `data/images/${filename}`;
      img.alt = filename;
      container.appendChild(img);
    });
  } catch (err) {
    console.error("Failed to load image list:", err);
  }
};

// await loadImagesClusterList();

const loadImagesClusterGrids = async () => {
  try {
    const res = await fetch("data/sortedImages_clusters.json");
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

    const clusters = await res.json();
    container.innerHTML = ""; // clear if needed

    // console.log(clusters);

    Object.entries(clusters).forEach(([clusterId, filenames]) => {
      // console.log(clusterId, filenames);
      const subGrid = document.createElement("div");
      subGrid.classList.add("sub-grid");

      filenames.forEach((filename) => {
        const img = document.createElement("img");
        img.src = `data/images/${filename}`;
        img.alt = filename;
        subGrid.appendChild(img);
      });

      container.appendChild(subGrid);
    });
  } catch (err) {
    console.error("Failed to load image list:", err);
  }
};

await loadImagesClusterGrids();
