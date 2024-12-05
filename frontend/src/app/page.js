"use client"

import TextStakeholderExtract from "@/components/TextStakeholderExtract";
import RLRunner from "@/components/RLRunner";
import FullPipeline from "@/components/FullPipeline";
import LLMInfoModal from "@/components/LLMInfoModal";
import { BlueButton } from '@/components/Classes'

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
      <div className='flex flex-col items-start w-full p-3'>
        {useFullPipeline ?
          <button onClick={() => setUseFullPipeline(false)} className={`${BlueButton}`}>Use Separate Components (LLM, RL)</button>
          :
          <button onClick={() => setUseFullPipeline(true)} className={`${BlueButton}`}>Use Full Pipeline</button>
        }
        <hr className='w-full my-3' />
        {useFullPipeline ?
          <FullPipeline />
          :
          <>
            <TextStakeholderExtract />
            <hr className='w-full my-3' />
            <RLRunner />
          </>
        }
      </div>
    </>
  );
}
