import axios from 'axios';

const API_BASE = 'https://unshaken-flatten-crabmeat.ngrok-free.dev';

export const uploadFiles = async (filesMap) => {
  const formData = new FormData();

  if (filesMap.device) {
    formData.append('device', filesMap.device);
  }

  if (filesMap.logon) {
    formData.append('logon', filesMap.logon);
  }

  if (filesMap.file) {
    formData.append('file', filesMap.file);
  }

  if (filesMap.email) {
    formData.append('email', filesMap.email);
  }

  if (filesMap.http) {
    formData.append('http', filesMap.http);
  }

  const res = await axios.post(
    `${API_BASE}/upload`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return res.data;
};

export const runModel = async () => {
  const res = await axios.post(`${API_BASE}/run-model`);
  return res.data;
};

export const getProgress = async () => {
  const res = await axios.get(`${API_BASE}/run-model/progress`);
  return res.data;
};

export const getUploadStatus = async () => {
  const res = await axios.get(`${API_BASE}/upload/status`);
  return res.data;
};