"use client"

import TextStakeholderExtract from "@/components/TextStakeholderExtract";
import RLRunner from "@/components/RLRunner";
import FullPipeline from "@/components/FullPipeline";
import LLMInfoModal from "@/components/LLMInfoModal";

import { useState } from "react";


export default function Home() {
  const [useFullPipeline, setUseFullPipeline] = useState(false);

  if (!process.env.NEXT_PUBLIC_RL_API_ENDPOINT) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <h1 className="text-3xl font-bold">RL API endpoint not set</h1>
        <p className="text-lg">Please set the RL API endpoint in a .env file</p>
      </div>
    )
  }

  return (
    <>
      <div className='flex flex-col items-start w-full'>
        {useFullPipeline ?
          <button onClick={() => setUseFullPipeline(false)} className="text-blue-500 underline">Use Separate Components</button>
          :
          <button onClick={() => setUseFullPipeline(true)} className="text-blue-500 underline">Use Full Pipeline</button>
        }
        <LLMInfoModal />
        {useFullPipeline ?
          <FullPipeline />
          :
          <>
            <TextStakeholderExtract />
            <RLRunner />
          </>
        }
      </div>
    </>
  );
}
