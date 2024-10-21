"use client"


import { useState } from "react";


export default function SampleInputAndResponse ({ className }) {
  const [prompt, setPrompt] = useState(null);
  const [response, setResponse] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    let input = e.target.input.value;
    setPrompt(input);

    // make api call
    const response = await fetch("/api/gpt", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ input })
    });
    const output = await response.text();

    setResponse(output);
  }

  return (
    <>
      <div className={className}>
        <form onSubmit={handleSubmit} className="flex flex-col items-start">
          <input name="input" placeholder="Input prompt here" required className="border-2 border-black" />
          <button type="submit" className="bg-green-500 hover:bg-green-600 active:bg-green-700">Submit</button>
        </form>

        <div className="font-bold">Query</div>
        {prompt ? <div>{prompt}</div> : <div className="text-gray-500">No input yet</div>}

        <div className="font-bold">Response</div>
        {response ? <div>{response}</div> : <div className="text-gray-500">No response yet</div>}
      </div>
    </>
  )
}