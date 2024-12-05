"use client"

import { useState } from "react";

import { ActiveButton, InactiveButton } from "@/components/Classes";

export default function RLRunner () {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [gifUrl, setGifUrl] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    let input = e.target.input.value;

    setLoading(true);
    
    // make api call
    const response = await fetch(process.env.NEXT_PUBLIC_RL_API_ENDPOINT, {
      method: "POST",
      body: input,
      headers: {
        "Content-Type": "text/plain"
      }
    });

    const output = await response.text();
    setResponse(output);

    // make api call for gif
    const gifResponse = await fetch(process.env.NEXT_PUBLIC_RL_API_ENDPOINT + "/gif", {
      method: "POST"
    });
    const blob = await gifResponse.blob();
    const url = URL.createObjectURL(blob);
    setGifUrl(url);

    setLoading(false);
  }

  return (
    <>
      <div className="font-bold text-lg">Reinforcement Learning</div>
      <div className='w-full'>
        <form onSubmit={handleSubmit} className="flex flex-col items-start">
          <textarea name="input" rows="5" placeholder="Input Stakeholder Features (CSV format)" required className="border-2 border-black w-1/2" />
          <div className="mt-1">
            <button type="submit" className={loading ? InactiveButton : ActiveButton} disabled={loading}>Submit</button>
          </div>
        </form>

        <div className="font-bold text-lg">Response</div>
        {loading ?
          <div className="text-gray-500">Running RL...</div>
          :
          (response && gifUrl ?
            <>
              <img src={gifUrl} alt="RL gif" className="w-1/2" />
              <div className="whitespace-pre-wrap">{response}</div>
            </>
            :
            <div className="text-gray-500">No RL response yet</div>
          )
        }
      </div>
    </>
  )
}