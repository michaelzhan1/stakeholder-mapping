import OpenAI from "openai";
const openai = new OpenAI();


// expect a json object
export async function POST(request) {
  const input = await request.text();

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "user",
        content: input
      },
    ],
  });

  let responseBody = completion.choices[0].message.content;

  return new Response(responseBody, {
    status: 200
  });
}