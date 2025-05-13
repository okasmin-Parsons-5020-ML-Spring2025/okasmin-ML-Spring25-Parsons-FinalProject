const container = document.getElementById("image-container");

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
// await loadImagesClusterGrids();

// Modal elements
const modal = document.getElementById("modal");
const modalContent = document.getElementById("modal-content");
const modalImage1 = document.getElementById("modal-image-1");
const modalImage2 = document.getElementById("modal-image-2");
const caption1 = document.getElementById("caption-1");
const caption2 = document.getElementById("caption-2");
const poster = document.getElementById("poster"); // Reference to the poster

// Function to open the modal
const openModal = (pair) => {
  // Set modal border color
  modal.style.borderColor = pair.color;

  // Set images and captions
  modalImage1.src = `data/images/${pair.objectId1}mask.png`;
  modalImage2.src = `data/images/${pair.objectId2}mask.png`;
  caption1.innerHTML = `<strong>${pair.department1}</strong><br>${pair.date1}`;
  caption2.innerHTML = `<strong>${pair.department2}</strong><br>${pair.date2}`;

  // Center modal within the poster
  const posterRect = poster.getBoundingClientRect();
  modal.style.top = `${posterRect.top + posterRect.height / 2}px`;
  modal.style.left = `${posterRect.left + posterRect.width / 2}px`;

  // Show modal
  modal.classList.add("active");
  poster.classList.add("modal-active");
};

// Function to close the modal
const closeModal = () => {
  modal.classList.remove("active");
  poster.classList.remove("modal-active");
};

// Add hover event listeners to all pair images
const addHoverListeners = () => {
  document.querySelectorAll(".grid-image").forEach((img) => {
    img.addEventListener("mouseenter", () => {
      const id = getObjIdFromFilename(img.src); // Extract object ID from filename
      const pair = pairs.find((p) => String(p.objectId1) === id || String(p.objectId2) === id);
      // console.log(pair);
      if (pair) openModal(pair);
    });
  });
};

// Close modal when clicking outside or pressing Escape
modal.addEventListener("click", closeModal);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

// found from process_images.ipynb and selected 3
const pairs = [
  {
    pair: "pair-1",
    objectId1: 47234,
    objectId2: 477748,
    distance: 0.037298230348537964,
    department1: "Asian Art",
    department2: "Medieval Art",
    date1: "first half of the 19th century",
    date2: "4th\u20137th century",
    color: "#d1fe59",
  },
  {
    pair: "pair-2",
    objectId1: 19903,
    objectId2: 244729,
    distance: 0.04808371645272125,
    department1: "The American Wing",
    department2: "Greek and Roman Art",
    date1: "ca. 1898\u20131918",
    date2: "ca. 1600\u20131050 BCE",
    color: "#8e56ff",
  },
  {
    pair: "pair-3",
    objectId1: 191213,
    objectId2: 48204,
    distance: 0.014794569606856212,
    department1: "European Sculpture and Decorative Arts",
    department2: "Asian Art",
    date1: "early 18th century",
    date2: "12th\u201313th century",
    color: "#8ee6ea",
  },
];

const getPairsForObjectId = (id) => {
  return pairs.filter(
    (pair) => String(pair.objectId1) === String(id) || String(pair.objectId2) === String(id)
  );
};

// await loadImagesClusterGridsBalanced();

function transformFixedGroups(obj, start) {
  const result = [];

  for (let i = 0; i < 5; i++) {
    // 5 chunks (40 items / 8)
    const chunk = [];

    for (let group = start; group < start + 5; group++) {
      // 5 groups
      const sliceStart = i * 8; // Renamed to avoid conflict
      const sliceEnd = sliceStart + 8;
      chunk.push(...obj[group].slice(sliceStart, sliceEnd));
    }

    result.push(...chunk);
  }

  return result;
}

const getObjIdFromFilename = (filename) => {
  const match = filename.match(/(\d+)(?=mask)/);
  return match ? match[1] : null;
};

const createBalancedClusterPosterGrid = async () => {
  const res = await fetch("data/sortedImages_clusters_balanced.json");
  const clusters = await res.json();

  const gridContainer = document.getElementById("grid-container");

  const sorted = [];

  for (let i = 0; i < 5; i++) {
    const chunk = transformFixedGroups(clusters, i * 5);
    sorted.push(...chunk);
  }

  sorted.forEach((filename) => {
    const id = getObjIdFromFilename(filename);

    const wrapper = document.createElement("div");
    wrapper.classList.add("grid-cell");

    const img = document.createElement("img");
    img.src = `data/images/${filename}`;
    img.alt = filename;
    img.classList.add("grid-image");

    const currPair = getPairsForObjectId(id);

    if (currPair.length > 0) {
      // console.log(currPair[0].pair);
      console.log(filename);
      img.classList.add(`${currPair[0].pair}`);
    }

    wrapper.appendChild(img);
    gridContainer.appendChild(wrapper);
  });

  // Add hover listeners after grid is created
  addHoverListeners();
};

await createBalancedClusterPosterGrid();

const loadImagesClusterGridsBalanced = async () => {
  try {
    const res = await fetch("data/sortedImages_clusters_balanced.json");
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
