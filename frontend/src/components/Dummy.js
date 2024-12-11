'use client'

import { useState } from "react";

import { BlueButton } from "./Classes";

export default function Dummy () {
  const [apiOutput, setApiOutput] = useState(null);

  const handleClick = async () => {
    const response = await fetch('http://localhost:5000/api/dummy', {
      method: "POST",
  });
    const output = await response.text();
    
    setApiOutput(output);
  }

  return (
    <>
      <button onClick={handleClick} className={`${BlueButton}`}>Test Flask API Call</button>
      <div>
        {apiOutput}
      </div>
    </>
  )
}