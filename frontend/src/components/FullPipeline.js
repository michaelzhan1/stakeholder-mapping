"use client"

import { useState } from "react";

import pdfToText from "react-pdftotext";

const POSITION_DEF = "Position (-1: Opposed, 0: Neutral, 1: Supportive): Position refers to the stakeholder's stance on the issue relative to the primary stakeholder, based on their actions.\n" +
"- Opposed (-1): The stakeholder actively works against the objectives of the primary stakeholder.\n" +
"- Neutral (0): The stakeholder does not take a definitive stance relative to the primary stakeholder.\n" +
"- Supportive (1): The stakeholder actively supports the objectives of the primary stakeholder.";

const POWER_DEF = "Power (0: Low, 1: Medium, 2: High): Power refers to the degree of control or influence a stakeholder has over the outcome of the negotiation. Sources of power may include legal authority, financial resources, political leverage, or social capital. Consider the stakeholder's access to these resources and how directly they can affect the final decision.\n" +
"- ⁠Low (0): The stakeholder has minimal or no control over the negotiation's outcome. They might be unable to mobilize resources or people effectively, have no formal authority, or be easily overruled. Even if they are engaged in the process, their influence is limited to providing input, with little to no impact on decisions.\n" +
"⁠- Medium (1): The stakeholder holds some influence over the negotiation but does not have decisive control. They may influence others' decisions or introduce changes, but their power is indirect or dependent on coalition-building. While they can sway opinions or set agendas, they are not the final decision-maker and require cooperation from others to effect change.\n" +
"⁠- High (2): The stakeholder has decisive control or a significant degree of leverage over the negotiation's outcome. They possess critical resources or authority and can dictate terms, veto decisions, or influence others to a degree that the negotiation cannot proceed effectively without their consent.";

const KNOWLEDGE_DEF = "Knowledge (0: Not Aware, 1: Limited, 2: Extensive): Knowledge refers to the stakeholder's level of awareness and actionable insights.\n" +
"- Not Aware (0): Lacks any awareness of the relevant issue.\n" +
"- Limited (1): Has some level of awareness but lacks sufficient information to effectively participate in or influence the negotiation.\n" +
"- Extensive (2): Possesses relevant insights and understanding that significantly affect the negotiation outcomes.";

const URGENCY_DEF = "Urgency (0: Low, 1: High): Urgency refers to the stakeholder's subjective perception of the time-sensitivity of the issue.\n" +
"- Low (0): The issue can be addressed on a flexible timeline without immediate consequences in the stakeholder's view.\n" +
"- High (1): The stakeholder perceives the issue as requiring immediate action to prevent significant negative outcomes (e.g., legal deadlines, financial losses, or operational disruptions).";

const LEGITIMACY_DEF = "Legitimacy (0: Not Legitimate, 1: Legitimate): Legitimacy refers to whether a stakeholder has the right or standing to be involved in the negotiation.\n" +
"- Not Legitimate (0): The stakeholder should not be involved in the negotiation.\n" +
"- Legitimate (1): The stakeholder has the relevance and standing to meaningfully participate in the negotiation.";

const EXTRACTION_PROMPT_CORE = "Given the following text data, identify the key stakeholders involved in the negotiation and infer their attributes based on the definitions provided below. You do not have prior knowledge of the ground truth values, so you must analyze the text and make educated estimates about the stakeholders' values for Power, Urgency, Knowledge, Position, and Legitimacy.\n" +
"\n" +
"For each stakeholder, determine the following attributes and present the values in CSV format. Limit to 5 stakeholders. Your prediction must be from the available options for each attribute, make your best guess if necessary. Do not use any markdown formatting, and only have the CSV data in raw text. Ensure a stakeholder name does not have any commas, for the sake of the CSV. Each row should look like the following:\n" +
"\n" +
"<Stakeholder name>,<Position: (-1, 0, or 1)>,<Power: (0, 1, or 2)>,<Urgency: (0 or 1)>,<Knowledge: (0, 1, or 2)>,<Legitimacy: (0 or 1)>\n" +
"\n" +
"Do not output any additional text. Use only the information implied in the text to make these determinations.\n" +
"\n" +
"Definitions:\n" +
`${POWER_DEF}\n` +
`${URGENCY_DEF}\n` +
`${KNOWLEDGE_DEF}\n` +
`${LEGITIMACY_DEF}\n` +
`${POSITION_DEF}\n` +
"\n";

export default function FullPipeline ({ className }) {
  const [usePdf, setUsePdf] = useState(false);
  const [llmLoading, setLlmLoading] = useState(false);
  const [rlLoading, setRlLoading] = useState(false);
  const [llmResponse, setLlmResponse] = useState(null);
  const [rlResponse, setRlResponse] = useState(null);

  const runRL = async (input) => {
    setRlLoading(true);

    const response = await fetch(process.env.NEXT_PUBLIC_RL_API_ENDPOINT, {
      method: "POST",
      body: input,
      headers: {
        "Content-Type": "text/plain"
      }
    });

    const output = await response.text();
    setRlResponse(output);

    setRlLoading(false);
  }

  const callGPT = async (textInput) => {
    setLlmLoading(true);

    // make api call for LLM extraction
    const response = await fetch("/api/gpt", {
      method: "POST",
      body: textInput
    });
    const output = await response.text();
    setLlmResponse(output);
    setLlmLoading(false);

    return output;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    let textInput = "";
    if (usePdf) {
      let file = e.target.file.files[0];
      const text = await pdfToText(file);
      textInput = EXTRACTION_PROMPT_CORE + `Text data:\n${text}`;
    } else {
      let input = e.target.input.value;
      textInput = EXTRACTION_PROMPT_CORE + `Text data:\n${input}`;
    }
    
    const gptOutput = await callGPT(textInput);
    runRL(gptOutput);
  }

  // Define button states
  const activeButtonClass = "bg-green-500 hover:bg-green-600 active:bg-green-700";
  const inactiveButtonClass = "bg-gray-300 cursor-not-allowed";

  return (
    <>
      <div className="font-bold text-lg">Stakeholder Extraction</div>
      <div className={className}>
        {/* TODO: popup modal that lets you view definitions*/}
        {usePdf ?
            <button onClick={() => setUsePdf(false)} className="text-blue-500 underline">Use text input</button>
            :
            <button onClick={() => setUsePdf(true)} className="text-blue-500 underline">Use PDF input (experimental)</button>
        }

        {usePdf ?
          <form onSubmit={handleSubmit} className="flex flex-col items-start">
            <input type="file" name="file" accept=".pdf" required className="" />
            <button type="submit" className={(rlLoading || llmLoading) ? inactiveButtonClass : activeButtonClass} disabled={rlLoading || llmLoading}>Submit PDF</button>
          </form>
          :
          <form onSubmit={handleSubmit} className="flex flex-col items-start">
            <textarea name="input" placeholder="Input prompt here" required className="border-2 border-black w-1/2" />
            <button type="submit" className={(rlLoading || llmLoading) ? inactiveButtonClass : activeButtonClass} disabled={rlLoading || llmLoading}>Submit</button>
          </form>
        }

        <div className="font-bold">LLM Response (limited to 5 stakeholders)</div>
        {llmLoading ?
          <div className="text-gray-500">Loading...</div>
          :
          (llmResponse ?
            <div className="whitespace-pre-wrap">{llmResponse}</div>
            :
            <div className="text-gray-500">No LLM response yet</div>
          )
        }
        
        <div className="font-bold">RL Response</div>
        {rlLoading ?
          <div className="text-gray-500">Running RL...</div>
          :
          (rlResponse ?
            <div className="whitespace-pre-wrap">{rlResponse}</div>
            :
            <div className="text-gray-500">No RL response yet</div>
          )
        }
      </div>
    </>
  )
}