# Manual error review (exp2_bilstm_attn)

### 1. confident_FP | test_idx 29330 | true=neg | P(pos)=1.000

> Wow love the place and everything is very clean and new!\n\nGreat place to come and relax worth a try!\n\nCheers,\n\nEric Van Nguyen\nVisited April 2012

**Error type:** label noise

**Proposed testable fix:** Have two people re-label the 50 most confident errors blind to the gold label; report the share that disagree as a noise ceiling on accuracy

### 2. confident_FP | test_idx 29525 | true=neg | P(pos)=1.000

> I tried to love it, had a chocolate chip cookie and milk once, and was satisfied.. took my date here after lunch next door and went all out on a vietnamese iced coffee ice cream and red velvet cookie sandwich.. the cookie was too hard and made it not enjoyable for me =(\n\nBut the ice cream was delicious !

**Error type:** mixed sentiment (praise + complaint)

**Proposed testable fix:** Concatenate max-pooled and last-state BiLSTM features with attention pooling; compare error on the contrast-word slice

### 3. confident_FP | test_idx 1560 | true=neg | P(pos)=1.000

> Home of the incredible exploding burrito. This place is awesome if you want to wear your food. Actually, that might be better than eating it. :(

**Error type:** sarcasm / irony (+ emoticon removed)

**Proposed testable fix:** Keep emoticons (:( :) =() as tokens instead of stripping them; compare error on reviews containing emoticons

### 4. confident_FP | test_idx 23815 | true=neg | P(pos)=1.000

> This is the neighborhood Foodland that has the bare necessities needed to sustain a pantry or for when the next snowstorm of the century is a day away and you only have minutes to get TP, bread and milk. \n\nThe bakery here is tops, small selection, but really well made specialities. Huge brownies, iced and moist. They are easily 5 inch squares,  topped with things like Oreo cookies, nuts, sprinkles, and other yummy delights. The best thing I found is the individual pineapple upside-down cakes. Perfect for one person, these are just like the big ones, same buttery cake, same crunchy edges from the brown sugar melting as it bakes, and that almost candied pineapple that has bece one of my favorite desserts of all time.\n\nTry the foccacia dripping with evoo, fresh basil and tomatoes and mozzarella cheese, you'll never buy pizzeria pizza ever again. Take a few home and pop them in the oven and enjoy. \n\nI come here when I need the staples of the pantry like eggs, milk, even a pound of ground beef. The deli isn't the greatest but it will do in a pinch. Good rotisserie chickens too.

**Error type:** mixed sentiment or label noise (check ending)

**Proposed testable fix:** Re-label blind (as above); if the ending is a complaint, test head+tail truncation on long reviews

### 5. confident_FP | test_idx 408 | true=neg | P(pos)=1.000

> Though I'm a Copper enthusiast when it comes to getting my Indian fix in Charlotte, I'd heard that Maharani was a cheaper but tasty option, so we ordered from there a few nights ago. \n\nCopper is definitely still my place, but Maharani was fine enough. First of all, the food came very quickly, which is rare. Usually indian food, good indian good, takes at least 30-45 minutes. We got our order in like 30, which was great 'cause we were starving. \n\nThe Tikka Masala was spicy and pretty good, but it wasn't as thick and saucy as i like. The salad was just so-so. The Naan were all amazing. Definitely the highlight.

**Error type:** lukewarm / comparative ('fine enough', 'still prefer X')

**Proposed testable fix:** Ensemble BiLSTM + Transformer probabilities; compare error in the 0.4-0.6 confidence band

### 6. confident_FN | test_idx 21813 | true=pos | P(pos)=0.000

> Our flight arrived to Vegas earlier than excepted, so we expected our room not to be ready. When we arrived at the hotel on May 19th, the front desk girl offered us a room that was ready on the 28th floor that wasn't facing the Bellagio fountain. I booked a corner view suite had requested a room on a high floor with a Bellagio fountain view when I first made the reservation. There are 56 floors at the hotel. She then said there was a room on the 30th floor with a view of the Bellagio fountain, but it wasn't ready. She continued to look for rooms, which she then said that there was a room on the 50th room with a Bellagio fountain view, but it wasn't ready yet either. We agreed to wait for that room to be cleaned, which she said it would be couple of hours before it would be ready. We waited a couple of hours by walking around the area before calling to check to see if the room was ready yet. I was transferred to housekeeping, who said it wasn't ready yet and they would call me back when it was. After eating and walking around some more for a couple more hours without hearing back from anyone, we decided to go back to the hotel and was told that the room was ready. It would of been n

**Error type:** truncation / late resolution (check ending)

**Proposed testable fix:** Use head+tail truncation (first 100 + last 100 tokens) or MAX_LEN=400; re-measure error on the long-review slice

### 7. confident_FN | test_idx 20061 | true=pos | P(pos)=0.000

> Ever wonder what to do if you have lots of extra garbage or recyclables and either can't fit them all in your bins or missed bulk trash pickup day? Alternatively, are you looking for something just a little bit different to do on a lazy summer day? \n\nOk, ok, all kidding aside - you may find yourselves (as we did yesterday) with more broken-down boxes and odd-shaped trash (old floor lamps, etc.) than can fit in your bins. This waste facility allows Phoenix residents to dump bulk trash and recyclables for free once a month. \n\nTo get to the facility, drive down to 27th Avenue and Buckeye (aka \""No Man's Land\""). When you drive up, you will likely see a long line of commercial vehicles waiting to do their daily or weekly dump. You can bypass this line and go to the R (the sign says something like \""Visitors\""). When I say \""you can,\"" what I mean is that we did without being stopped, so you should be able to as well. According to the city website, we were supposed to have to show a driver's license and recent water bill to be let in, but no one stopped us or asked for proof of anything. \n\nWe drove around and around the facility, following signs that directed us to the visit

**Error type:** topic-sentiment confound + truncation (humour about a trash facility)

**Proposed testable fix:** Head+tail truncation and a larger MAX_LEN; re-measure long-review error

### 8. confident_FN | test_idx 22807 | true=pos | P(pos)=0.000

> EDIT: They really did change the service up since I last posted this.\n\nHorrible service.\n\nUsed to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when we were paying with cash. He then proceeded to disrespect the party at the table, telling us to 'not give him attitude about it.'\n\nSorry Bella Notte, but we're not children. I don't care if you're working hard - it doesn't give you any excuse to disrespect your paying customers like that.

**Error type:** label noise (text is clearly negative; 'EDIT' line suggests a stale rating)

**Proposed testable fix:** Re-label blind; measure error on reviews containing 'EDIT' or 'UPDATE'

### 9. confident_FN | test_idx 21215 | true=pos | P(pos)=0.000

> Last night several parents came in with over 15 children to celebrate their 9 year olds 4th grade graduation at 9:15 pm. The bartender expressed that it was not a place to have children running around as it is against the law and a liability issue if anything were to happen to them on their premise. The children were running in and out of the bar while the parents continued to drink upstairs claiming to watch them in the open park way. After their third round of drinks the bartender told them they were no longer welcomed due to the fact the kids were unsupervised as well as the other customers had cleared out expressing \""whether the bar was a childcare center\"". One father even started swearing at her telling her it was none of her business what their children were doing. I am offended in that they completely took advantage of her, were rude, not to mention had their children with them at a bar at 10:00pm in the evening. I will say the bartender was very cordial and had mind to do the right thing.

**Error type:** reported events / third-party narrative (check ending)

**Proposed testable fix:** Add last-N-token pooling to the BiLSTM; compare error on reviews with quotes or reported speech

### 10. confident_FN | test_idx 16700 | true=pos | P(pos)=0.000

> TERRIBLE SERVICE, RUDE WAITERS WITH A PISS POOR ATTITUDE! WOULD EAT HERE AGAIN! A++++\n\nThe food here is good but isn't that spectacular, the beer selection is somewhat disappointing if you like a good craft beer. Ask for a craft beer and get an insulting comment from the waitress. \n\nWhat makes this place awesome is the atmosphere, live bands, sports on the TVs, the very obnoxious staff and the hats. Got to love the free hats. \n\nIf you go, ask for a glass of water with your meal. ;)

**Error type:** sarcasm / irony (ALL-CAPS complaints, 'A++++')

**Proposed testable fix:** Stop lowercasing blindly: add an ALL-CAPS ratio feature and keep tokens like 'a++++'; compare error on caps-heavy reviews

### 11. near_threshold | test_idx 28040 | true=pos | P(pos)=0.499

> Great beer selection, especially the beer fusions. Nice atmosphere, friendly staff, quick service. Food is typical bar food, nothing special but didn't agree with my girlfriends stomach in the least. So four stars for the beer and atmosphere but wouldn't eat here.

**Error type:** mixed sentiment (explicit star verdict)

**Proposed testable fix:** Ensemble BiLSTM + Transformer; compare accuracy on reviews with p in 0.4-0.6

### 12. near_threshold | test_idx 9912 | true=neg | P(pos)=0.502

> Stayed at TI for the first time for my 50th birthday. Booked online at TI's website and the deal seemed really good for a petite suite. Our stay: \n\nRoom:\nThe room was decent but nothing spectacular. Other than a bit more space, the biggest difference from most standard rooms was having two full bathrooms; one with a shower and one with a tub. The bed and pillows were also decent/fairly comfortable. On the negative side the tv was pretty small for a suite, I think it was only 32 inches. Also, and despite being on the 27th floor (mountain view) there was still a considerable amount of noise from the roadway below. Although the room had a refrigerator, it was empty, no minibar items, no snacks, no water bottles (for purchase or otherwise). We were told at check out that too many guests were having problems with the touch sensitive items in the fridge (making unintended purchases) that they removed the minibars all together. \n\nFood\nOur deal included 2 for 1 buffet. We ate there on a Sunday for the champagne brunch. The regular price is $23.95 and despite having the 2 for 1 deal, we didn't feel it was worth the cost. Food selection was pretty limited for a Sunday champagne brunch;

**Error type:** mixed sentiment + truncation (570 words)

**Proposed testable fix:** Head+tail truncation; compare error on reviews longer than MAX_LEN

### 13. near_threshold | test_idx 14409 | true=pos | P(pos)=0.497

> Awesome Thai food here!! My favorite is the yellow curry.  We eat here a few times a month.  I will agree that the service isnt the best or the fastest but sometimes you can manage to get in and out in a decent amount of time.  We recently ate here for lunch and the curry wasnt up to par, so we pointed it out to one of the waitstaff, she offered to bring a new plate of it, but its all cooked together in a big pot, so that didnt really make sense. Then the owner, who is the cook came out and TOLD me that it is the same as it always is, and was actually irritated that I would even question her cooking.  The friendly waitress that offered a new plate even said to us that the owner was **** hurt that a customer wasnt happy.  If I owned a place, and my food was off a day, I would want a nice customer to carefully point it out and correct it, not argue that its how it should be.\nOf course we'll eat here again and again and again, because we have had many great meals here.  I just wish that owners and staff would recognize in this tough economy, that they should take care of their customers, and treat them kindly and recognize that sometimes people need to have quicker service because th

**Error type:** mixed sentiment (praise first, complaint later)

**Proposed testable fix:** Ensemble BiLSTM + Transformer; compare error in the 0.4-0.6 band

### 14. near_threshold | test_idx 7242 | true=neg | P(pos)=0.503

> Visiting old part of vegas during their Motor Bike Fest was a bad idea lol...Jammed packed line up past the door, Tony Roma's was the food place to be! located across from the Golden Nugget.\nThe service was an overall 2/5, there was mass confusion at first as we had three different servers coming to us.\n\nSampler platter 3.5/5\nPotato skins with bacon, cheddar was delightful.  The wings were nice and spicy!!! and their mozarella sticks were pretty yum.\n\nPrime Rib: $8.95 special 3/5 which included baked potatos and veggies.  Presentation was there, however not the best prepared...\n\noverall this place was grungy, and quite noisy. Though this place wont break your wallet in terms of price, but you;ll have to sacrifice the service and quality.  HOWEVER this place was packed....

**Error type:** numeric ratings in text ('2/5', '3.5/5') destroyed by cleaning

**Proposed testable fix:** Map patterns like 'x/5' to rating tokens (<rating_low>, <rating_high>); compare error on reviews containing 'x/5'

### 15. near_threshold | test_idx 3822 | true=neg | P(pos)=0.503

> What a shame...this store is closing after 21 years in business.  The family that owns the Red Rooster is moving to Illinois.  The lady that runs the place was very friendly and told me this after I inquired about the closing 7/15/10 sign on the door.\n\nThey used to have multiple floors and many rooms.  Now, they are reduced to a garage sale amount of stuff that appears to have been picked over.\n\nI am sure this was a 4 or 5-star place at one time, however that day has passed.  The good news is that the Charleston Antique Mall is in full swing literally next door.  Go there instead.

**Error type:** implicit negative / past-tense praise ('was a 5-star place at one time')

**Proposed testable fix:** Remove 'was', 'used', 'once', 'anymore' from the stopword list; compare error on reviews with 'used to', 'anymore', 'at one time'

### 16. slice: has contrast (but/however/although) | test_idx 11908 | true=pos | P(pos)=0.124

> Return visit for work.  I was training a new hire, and remembered this great little shop. Not a good place to study:/ or have a work conversation. Too much noise between the 90s music and staff... Still great coffee but not a place to concentrate:).

**Error type:** contrast: final clause decisive ('Still great coffee') + emoticons stripped

**Proposed testable fix:** Keep emoticons and add last-state pooling; compare contrast-slice error

### 17. slice: has contrast (but/however/although) | test_idx 23417 | true=pos | P(pos)=0.255

> Brilliant idea. Put a roasted pig in the window and feed him/her to the public. When the meat of this pig is gone, close the shop. \nAll they sell is this pig all day.\nWhat a wonderful concept, so simple and soooooooooooooooooo delicious. The taste is more like American style pulled-pork BBQ, though without the smokiness or BBQ sauce. With this, you get sage and onion seasoning sprinkled on it, and perhaps some haggis or sweet apple sauce on top. I live in Germany now, and this would be the only place outside of Germany in Europe that would make me forsake a Bratwurst.\n\n\n Only caveat--- the bread was cheap, horrible hamburger buns straight out of the supermarket aisle. Yuck! They need decent bread, something as hearty and heavy to stand up to the pork. Some artisan, handmade bread would complement the handmade meat, instead of being a cheap-ass afterthought. I deducted 1 star for the horrible bread rolls.

**Error type:** elongated words (OOV) + trailing caveat

**Proposed testable fix:** Collapse letters repeated 3+ times ('soooo' -> 'so'); compare error on reviews containing elongated words

### 18. slice: has contrast (but/however/although) | test_idx 16460 | true=neg | P(pos)=0.823

> Do you come here often? This old line would definitely work here. Great for families or old marrieds looking to get out of the house and laugh-out-loud for a few hours. This place is a brightly colored, entertainingly, loud, home away from home, covered in unexpected but, laughtasticly humorous saying that creatively remind you of where you are.\n\nI have been here a few times in the past, with friends, unsually during the day (Day drinking, as their shirts say). So, the other Saturday night I decided to head over after some strenuous solo christmas shopping. Looking to relax, refuel, drink, flirt and..... (nevermind, it didn't happen anyways). I parked near the Greenfield side and walked up to what I thought was the main door. But, it was locked! The thought crossed my mind that they may be closed? But, the loud music led me around the corner to the chill-friendly security door check who informed me that they serve food until 2am and wished me a good time. As I made my way inside the place was pretty busy on the patio, with a live DJ, no seats left at the bar, and absolutley no one but staff on the inside, I guess that was their intention.\n\nI found a nice spot (with great viewin

**Error type:** long mixed review (658 words) + truncation

**Proposed testable fix:** Head+tail truncation; compare long-review error

### 19. slice: has contrast (but/however/although) | test_idx 13040 | true=neg | P(pos)=0.847

> I cant believe I ever went here, but my girlfriend did the models hair that were doing a fashion show for the ugliest homemade garbage. So I came for support. \n\nAnyway, every dude in there looks the same. Low rise seven jeans with a sparkley belt with shirt tucked behind the questionably femenine buckle, sunglasses inside, orange tan, waxed eyebrows, and shaved arms.\nI spoted the  lone creepy old dude that was obviously on raver drugs, in the middle of the dance floor raving out. \nBros and sluts were all freek dancing and being douchy and slutty.\nA ton of younger chicks with older dudes.\n\nIts one of those places were dumb sluts starved for attention come and make out with each other in front of the ed hardy faux buff guys. \n\n(faux buff= fat that looks like muscle under a shirt)\n\nWhats up with all the faux buff dudes in Scottsdale. You know what Im talking about.\n\n\nI dont know why they would ever charge ten dollars to come here.

**Error type:** slang / misspellings / profanity (check)

**Proposed testable fix:** Use subword (BPE) tokens trained on this corpus; compare OOV rate and error on high-OOV reviews

### 20. slice: has contrast (but/however/although) | test_idx 1253 | true=neg | P(pos)=0.870

> This location certainly draws a large lunch crowd.  I was there with my son yesterday at about 12:30 and the place was packed.  The staff seemed to be having a hard time keeping up with everything. For instance, there were about 6 plates...more than we needed.  I did see some setting to the side that had leftover food residue from the last guest sitting to the side...I'm not counting those.  Anyhow, behind my son and I was a group of 6, so we weren't all eating right away!  I have been to this location in the past and there were no napkins at all - they used papertowels from the bathroom.  You pulled off what you needed and went with it.  Yikes!  Where is management when you need them?\nFood quality was good.  Veggies were fresh, some of the quantities pretty skimpy.  I understand it was lunch time.\nIt seemed there were 2 servers only, and they were hustling. More were certainly needed.\nOverall, it's a good place to go.  I do enjoy meeting a friend of mine here on an occaisional Sunday afternoon when it's much calmer...altho that's when we had the paper towels issue...but, service is friendly, food is pretty good.\nOh...and yesterday, the best soup...Chicken Enchilada Soup...if i

**Error type:** implicit negative (event descriptions, few sentiment words)

**Proposed testable fix:** Train a 2-layer BiLSTM; compare error on the contrast slice and overall

