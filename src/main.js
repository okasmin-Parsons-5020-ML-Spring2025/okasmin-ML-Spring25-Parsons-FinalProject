const container = document.getElementById("image-container");

const loadImages = async () => {
  try {
    const res = await fetch("data/sortedImages.json");
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

    const filenames = await res.json();

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

await loadImages();
