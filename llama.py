import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

# pip install --upgrade transformers
torch.set_grad_enabled(False)

model_names = [
    # "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B",
]
model_names_plot = [
    # "Llama-2-7B",
    # "Llama-3-8B",
    "Llama-3.1-8B",
    "Llama-3.1-70B",
]


def calculate_per_token_perplexity(model, tokenizer, input_string, k):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")

    perplexities = []
    num_tokens = input_ids.size(1) - 1

    for i in range(k, num_tokens):
        current_input = input_ids[:, : i + 1]
        target = input_ids[:, i + 1]

        with torch.no_grad():
            outputs = model(current_input)
            logits = outputs.logits

        loss = torch.nn.functional.cross_entropy(logits[:, -1, :], target)
        # perplexity = torch.exp(loss).item()
        perplexities.append(loss.item())

    return perplexities


input_string = "We call our particular attention “Scaled Dot-Product Attention” (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by the square root of dk, and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V. We compute the matrix of outputs as: Attention(Q, K, V) = softmax(QK^T / square root of dk) V The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1 / square root of dk. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code. While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot-product attention without scaling for larger values of dk. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by 1 / square root of dk."
# input_string = "But the problem, you see, when you ask why something happens, how does a person answer why something happens? For example, Aunt Minnie is in the hospital. Why? Because she went out, slipped on the ice, and broke her hip. That satisfies people. It satisfies, but it wouldn't satisfy someone who came from another planet and who knew nothing about why when you break your hip do you go to the hospital. How do you get to the hospital when the hip is broken? Well, because her husband, seeing that her hip was broken, called the hospital up and sent somebody to get her. All that is understood by people. And when you explain a why, you have to be in some framework that you allow something to be true. Otherwise, you're perpetually asking why. Why did the husband call up the hospital? Because the husband is interested in his wife's welfare. Not always, some husbands aren't interested in their wives' welfare when they're drunk, and they're angry. And you begin to get a very interesting understanding of the world and all its complications. If you try to follow anything up, you go deeper and deeper in various directions. For example, if you go, Why did she slip on the ice? Well, ice is slippery. Everybody knows that, no problem. But you ask why is ice slippery? That's kinda curious. Ice is extremely slippery. It's very interesting. You say, how does it work? You could either say, I'm satisfied that you've answered me. Ice is slippery; that explains it, or you could go on and say, Why is ice slippery? and then you're involved with something, because there aren't many things as slippery as ice. "
# input_string = "Yeah. I find it a little bit more tightly, right? So let’s go back to desire, right? This is old, old Buddhist wisdom. I’m not saying anything original. But desire, to me, is a contract that you make with yourself to be unhappy until you get what you want. Okay? And I keep that in front of minds. So when I’m unhappy about something, I look for what is the underlying desire that I have that’s not being fulfilled. It’s okay to have desires. You’re a biological creature, and you’re put on this earth, you have to do something, you have to have desires, you have a mission. Don’t pick them up unconsciously. Don’t pick them up randomly. Don’t have thousands of them. My coffee’s too cold, doesn’t taste quite right. I’m not sitting perfectly. Oh, I wish it was warmer. You know. My dog, you know, pooped in the lawn, I don’t like that. Whatever it is. Pick your one overwhelming desire. It’s okay to suffer over that one, but on all the others, do you want to let them go so you can be calm and peaceful and relaxed? And then you’ll perform a better job. Most people, when you’re unhappy, like a depressed person, it’s not that they have very clear calm mind. They’re too busy in their mind. Their sense of self is too strong. They’re sitting indoors all the time. Their minds working, working, working. They’re thinking too much. Well, if you want to be a high performance athlete, how good of an athlete are you going to be if you’re always having epileptic seizures? If you’re always like twitching and running around and like jumping, and your limbs are flailing out of control?"
# input_string = "I mean, here’s the bad news — You’re gonna die. Here’s the good news — When you get closer, you’re gonna want to fucking die. It doesn’t keep being good. You don’t get to keep your life the way it is. Like, I’m 55. Both hands hurt. Both hands. Both my hips hurt. I’m lucky I have only one asshole. The whole fucking body hurts. When you’re, like, in your 30s, you don’t even think about it. You’re like — ‘Cause here’s, like — There’s the beginning of your life, and there’s the end. So, like, you start, like, in your 3– Like, you’re in your 30s. You’re just shuffling. You’re not even aware of this movement. You’re just shuffling along. “Hey. Life is good. Pretty cool.” And then somewhere you’re kind of aware that way down there, people are — Aah! — falling off the edge. Aah! Oh. Yeah. Weird. And then you’re not thinking. Then all of a sudden, you’re close, and you’re, Aah! Oh, fuck, I knew that guy. Oh, my God. Holy shit. But there are signs that your time is coming to an end, you know? Like — Like they don’t make my shampoo anymore. They don’t make it. It’s like, why’d you stop? Like, you’re the only one. Nobody fucking cares about you. And so I thought, I’m like, I’m gonna make my own. I’m gonna make my own shampoo. I’m gonna look it up online. I’m gonna look at the ingredients. I’m gonna make my own. And I found myself at a dinner telling a table full of people, “So I decided to make my own shampoo. That’s one of those moments where you tell yourself, Just don’t tell folks anything. You don’t need to say it out loud. Just do what you’re doing. Die as soon as possible."
# input_string = "Now, what's amazing about the olfactory neurons is that they are among the very few neurons in the human and other mammalian nervous system that regenerates throughout the lifespan. So there's a little area of your hippocampus where there's some neurons that everyone makes a big deal of that, frankly, don't do a lot to regenerate throughout the lifespan. So called neurogenesis, new neurons. But the olfactory neurons, even though they're a central nervous system neuron, just like your retinal neuron or your cerebral cortex, they can regenerate throughout the entire lifespan, and they do every time somebody takes a head hit or there's some, you know, shearing off of these axons, excuse me, they regenerate. Now, under conditions like, we saw this a lot during COVID where people were complaining about loss of smell. We see this when people age. Some people are thinking that loss of smell may be a correlate, not the cause, but obviously, but a correlate of age related cognitive decline, dementia and Alzheimer's, things like that. There are a few things, actually. I think I recommend it to a couple of friends of ours. Now, there's very little data on this, but I will say, and I'll catch heat for this, but these days, I catch heat anyway, so I don't care."
# input_string = "I just don't think it makes any sense, because if you take the multiverse literally. So let me back up a second. We have a principle that tells us more or less what is true, called parsimony, right? We take the simplest explanation that accounts for what we observe, and we imagine it's true. And there's a little imperfection in there. But if you had all the information, it would work, I think, perfectly. And then there's a flaw in how we apply it. The multiverse is analytically very simple. Right. It's just one move. Oh. There are an infinite number of universes. Every moment, there are an infinite number of things that could happen, and a universe is created for each one. That's very simple. I just said it. And one sentence. On the other hand, at the practical level, it couldn't possibly be more wasteful and absurd. Right? And the idea that there's going to be a different. There's going to be two universes. You're going to double the universe because I just moved my glasses, and we need one universe in which I didn't and one universe in which I did, and then each of those universes is going to proliferate out from each moment."
# input_string = "There’s chaos and potential, and that would be like the potential you should live up to, because everyone says, well, you should live up to your potential. It’s like, what the hell’s that? You can’t measure it or touch it or taste it or feel it. It’s this hypothetical thing that everyone regards as real. It’s like the future; what’s the future? Well, it’s not here yet, you can’t measure it. What makes you think it’s real? Well, we act as if it’s real and that seems to work. There’s the— so there’s potential, that’s one, that’s chaos, chaotic potential. Then there’s order, and that’s the structure that you need in order to confront the chaos. And you’d be born with that biologically, and then there’s your ability to call forth from the potential, new order. And that’s what you do with your speech. And that’s what happens in the first chapter of genesis it that God uses— God orders, let’s say— uses the power of truthful speech, that’s the logos, to transform potential into order and that’s what people are made in the image of. So there’s this theory, it’s a lovely theory that’s laid out right at the beginning of the bible that says that, if you tell the truth you transform the potential of being into a habitable actuality, that’s how it works. Say, how do you— how do you make the world better? Tell the truth. Because the world you bring into being as a consequence of telling the truth will be a good world. And I believe that’s true. I think it’s true metaphorically. I think it’s true theologically, and I think it’s true at the practical and scientific level as well. I think it’s true on all those levels simultaneously. So that’s being ridiculously exciting to sort through."
# input_string = "I said there was blood. I had more blood. I didn't know I had that much blood. The doctors later told me that the ear is a place that is a very bloody place if you're going to get hit. But in this case, it was probably the best alternative you could even think about, because it went at the right angle, and it was a hard hit. It was very, I guess you say surreal, but it wasn't surreal. I was telling somebody, you have instances like this or a lot less than this, where you feel it's a surreal situation. And I never felt that way. I knew immediately that it was a bullet. I knew immediately that it was at the ear, and because it hit very hard, but it hit the ear."
# input_string = "We have therefore wanted to say that all our intuition is nothing but the representation of appearance; that the things that we intuit are not in themselves what we intuit them to be, nor are their relations so constituted in themselves as they appear to us; and that if we remove our own subject or even only the subjective constitution of the senses in general, then all constitution, all relations of objects in space and time, indeed space and time themselves would disappear, and as appearances they cannot exist in themselves, but only in us. What may be the case with objects in themselves and abstracted from all this receptivity of our sensibility remains entirely unknown to us. We are acquainted with nothing except our way of perceiving them, which is peculiar to us, and which therefore does not necessarily pertain to every being, though to be sure it pertains to every human being. We are concerned solely with this. Space and time are its pure forms, sensation in general its matter. We can cognize only the former a priori, i.e., prior to all actual perception, and they are therefore called pure intuition; the latter, however, is that in our cognition that is responsible for its being called a posteriori cognition, i.e., empirical intuition. The former adheres to our sensibility absolutely necessarily, whatever sort of sensations we may have; the latter can be very different."

k = 10  # Start calculating perplexity from the 10th token
all_perplexities = {}

for model_name in model_names:
    print(f"Processing model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/home/aharona/ddn/isilon-migrated/Datasets/Aharon/llama_models",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        cache_dir="/home/aharona/ddn/isilon-migrated/Datasets/Aharon/llama_models",
    )

    perplexities = calculate_per_token_perplexity(model, tokenizer, input_string, k)
    all_perplexities[model_name] = perplexities

    avg_perplexity = sum(perplexities) / len(perplexities)
    print(f"Average Perplexity: {avg_perplexity:.4f}")

    del model, tokenizer
    torch.cuda.empty_cache()

# Calculate mean and std for each model
mean_perplexities = [np.mean(all_perplexities[model]) for model in model_names]
std_perplexities = [np.std(all_perplexities[model]) for model in model_names]
# Create the combined figure with 3 subfigures
plt.figure(figsize=(15, 12))

# Subfigure 1: Bar graph
plt.subplot(3, 1, 1)
plt.bar(model_names_plot, mean_perplexities, yerr=std_perplexities, capsize=5)
plt.ylabel("log Perplexity")
plt.title("Mean log Perplexity for LLaMA Models")

# Subfigure 2: Scatter plot
plt.subplot(3, 1, 2)
plt.scatter(
    all_perplexities["meta-llama/Meta-Llama-3.1-8B"],
    all_perplexities["meta-llama/Meta-Llama-3.1-70B"],
    alpha=0.5,
)
plt.xlabel("log Perplexity of Llama-3.1-8B")
plt.ylabel("log Perplexity of Llama-3.1-70B")
plt.title("log Perplexity Comparison: 8B vs 70B LLaMA Models")
# Add x=y line
max_perplexity = max(
    max(all_perplexities["meta-llama/Meta-Llama-3.1-8B"]),
    max(all_perplexities["meta-llama/Meta-Llama-3.1-70B"]),
)
plt.plot([0, max_perplexity], [0, max_perplexity], "r--", label="x=y")
plt.legend()

# Subfigure 3: Input text display
plt.subplot(3, 1, 3)
plt.text(0.5, 0.5, input_string, ha="center", va="center", wrap=True, fontsize=10)
plt.axis("off")

# Save the combined figure
plt.tight_layout()
plt.savefig("llama_models_combined_figure.png", dpi=300, bbox_inches="tight")
plt.close()
