Me ChatGPT the world's dummest AI (average incompetent) that does not understand anything generated this text vomit, for you to evaluate:

# ANNOTATION_INSTRUCTIONS.md

## Purpose

For each input `.txt` file, read the OCR-extracted receipt or order-related text and convert it into a corrected, normalized **JSON-LD** document using **Schema.org**.

The output must represent the document primarily as a `schema:Order` whenever the text is fundamentally a confirmation or record of a transaction. This matches the Schema.org definition of `Order`, which is a confirmation of a transaction (a receipt) and can contain multiple accepted offers. JSON-LD is a JSON-based format for linked data and supports either a single top-level object or linked entities via `@graph`. ([Schema.org][1])

Your job is not transcription. Your job is **semantic normalization** of noisy OCR text into the most accurate, receipt-grounded, non-hallucinated structured data possible.

---

## Required output format

Output must be:

* valid JSON
* valid JSON-LD
* only the JSON-LD object
* no markdown fences
* no explanations
* no commentary
* no citations inside the JSON-LD
* no debug notes
* no provenance/audit section unless explicitly requested

Use:

* `"@context": "https://schema.org"`
* either:

  * a single top-level object, or
  * `"@graph"` when multiple linked entities are needed

Use stable local `@id` references such as:

* `"#order"`
* `"#merchant"`
* `"#customer"`
* `"#delivery"`
* `"#offer-1"`
* `"#order-item-1"`
* `"#product-1"`

---

## Core annotation objective

Convert noisy OCR receipt text into the most semantically correct **Schema.org Order-centered JSON-LD** representation possible.

The model must learn all of the following behaviors:

* general broken values may be corrected when strongly supported by the receipt text itself
* unique or sensitive specific values must not be guessed
* intact unique/sensitive values should be copied faithfully
* broken unique/sensitive values must be represented with typed machine-readable error markers instead of invented corrections
* absent facts must not be added
* ambiguous facts must be omitted or marked unresolved rather than hallucinated

---

## Primary type rule

Treat the document as an `Order` by default when it is any of the following:

* a receipt
* an order confirmation
* a purchase confirmation
* a paid order summary
* a checkout confirmation
* a shipped-order summary that still clearly describes the purchase
* a pickup-ready order notice
* a service purchase confirmation
* a digital purchase confirmation

Do **not** switch away from `Order` merely because the document is sparse, ugly, or OCR-damaged.

The top-level object should usually be:

```json
{
  "@context": "https://schema.org",
  "@type": "Order"
}
```

or an `@graph` containing an `Order` node.

---

## Order-related Schema.org types you may use

Use Schema.org objects wherever they fit naturally. Do not flatten everything to strings when a better semantic structure exists.

### Commonly useful order-related types

Use these when appropriate:

* `Order`
* `OrderItem`
* `Offer`
* `Product`
* `Service`
* `Organization`
* `LocalBusiness`
* `Store`
* `Person`
* `PostalAddress`
* `QuantitativeValue`
* `PriceSpecification`
* `PropertyValue`
* `ParcelDelivery`

These are all official Schema.org types or values relevant to order/receipt modeling. `OrderItem` is for line items in an order, `acceptedOffer` is a property on `Order`, `orderDelivery` links order delivery information, `ParcelDelivery` models delivery state, `PropertyValue` is the generic property/value type, and `PaymentComplete` is the payment status meaning payment has been received and processed. ([Schema.org][1])

### Typical role mapping

Use the most semantically correct structure available:

* merchant or seller:

  * `Store`
  * `LocalBusiness`
  * `Organization`

* address:

  * `PostalAddress`

* purchased line:

  * `Offer`
  * optionally also `OrderItem`

* purchased thing:

  * `Product`
  * or `Service`

* quantity:

  * `QuantitativeValue`

* leftover labeled fact with no better schema property:

  * `PropertyValue`

* shipping/delivery information:

  * `ParcelDelivery`

* totals, subtotal, discounts, taxes, rounding, tendered amount, change:

  * scalar properties where appropriate
  * or `PriceSpecification` when a structured price object is clearly useful

---

## Strong grounding rule

Every important value must be grounded in the receipt text itself.

Do not invent facts.

Do not enrich the output using outside knowledge unless the receipt itself already supports a specific candidate reading and outside information is used only to validate that candidate.

The annotation target is **receipt-grounded structure**, not enrichment.

---

## Correction rule

You may correct OCR noise only when the correction is strongly supported by the input itself.

You must not silently “clean up” specific values from intuition alone.

### Examples of values that may be corrected when strongly supported

These are often general or semi-general values:

* date formats
* time formats
* currency codes
* decimal separators
* common payment words
* obvious total/subtotal/tax labels
* obvious product words when the receipt clearly supports the correction
* obvious generic store words such as `MARKET`, `STORE`, `TOTAL`, `CASH`, `VAT`, when the OCR damage is trivial

### Examples of values that must not be guessed

These are often specific, unique, personal, or identifier-like:

* phone numbers
* merchant names when uncertain
* company names
* company registration numbers
* VAT numbers
* tax IDs
* receipt numbers
* order numbers
* invoice/document numbers
* transaction IDs
* member IDs
* loyalty numbers
* customer names
* cashier names
* account identifiers
* card tail numbers when unclear
* street addresses when unclear
* URLs
* emails
* GTIN / barcode numbers when unclear

If such a value is broken and cannot be recovered with high confidence from the receipt itself, do **not** guess it.

---

## Sensitive and unique value rule

Sensitive, reviewer-relevant, personal, or unique values must be handled more strictly than general text.

Use this decision rule:

### 1. Specific value is present and clearly readable

Copy it faithfully.

### 2. Specific value is present but broken or noisy, and cannot be repaired with high confidence

Do not guess it. Use a typed error marker.

### 3. Specific value is not present at all

Do not invent it.

### 4. General value is broken but clearly recoverable from local receipt context

Normalize it.

This is one of the main supervision goals.

---

## Broken-value error code rule

When a clearly identified field contains a **specific** value that is broken, noisy, partially unreadable, or otherwise not reliably recoverable, output the field in a machine-readable unresolved form using an explicit error code.

This is required so the model learns:

* correct general values when grounded
* preserve intact specific values
* do not hallucinate broken specific values
* emit typed failure states for broken specific values

### Required pattern for broken specific values

Use one of these two patterns.

### Preferred pattern: structured field with empty value plus error code

Use this when the field identity is clear and a native field or `PropertyValue` slot exists.

Example:

```json
{
  "@type": "PropertyValue",
  "name": "merchantPhone",
  "value": "",
  "description": "ERROR_CODE:BROKEN_PHONE_NUMBER"
}
```

This preserves the fact that the field exists while making the value unresolved.

### Alternative pattern: native field plus companion `additionalProperty`

When a native Schema.org property should still exist, keep the property empty only if that property is naturally string-like and the field identity is clear, then attach the error code separately.

Example:

```json
{
  "telephone": "",
  "additionalProperty": [
    {
      "@type": "PropertyValue",
      "name": "telephone",
      "value": "",
      "description": "ERROR_CODE:BROKEN_PHONE_NUMBER"
    }
  ]
}
```

Prefer the first pattern unless there is a strong reason to preserve the native property directly.

---

## Standard error code vocabulary

Use uppercase snake case.

Use the narrowest correct error code you can justify.

### Merchant / company / identity errors

* `BROKEN_COMPANY_NAME`
* `BROKEN_MERCHANT_NAME`
* `BROKEN_BRANCH_NAME`
* `BROKEN_COMPANY_NUMBER`
* `BROKEN_VAT_ID`
* `BROKEN_TAX_ID`
* `BROKEN_BUSINESS_ID`

### Contact errors

* `BROKEN_PHONE_NUMBER`
* `BROKEN_EMAIL`
* `BROKEN_URL`

### Person-related errors

* `BROKEN_PERSON_NAME`
* `BROKEN_CUSTOMER_NAME`
* `BROKEN_CASHIER_NAME`
* `BROKEN_MEMBER_NAME`

### Identifier errors

* `BROKEN_ORDER_NUMBER`
* `BROKEN_RECEIPT_NUMBER`
* `BROKEN_DOCUMENT_NUMBER`
* `BROKEN_TRANSACTION_ID`
* `BROKEN_MEMBER_ID`
* `BROKEN_LOYALTY_ID`
* `BROKEN_ACCOUNT_ID`
* `BROKEN_CARD_LAST4`
* `BROKEN_BARCODE`
* `BROKEN_GTIN`

### Address errors

* `BROKEN_STREET_ADDRESS`
* `BROKEN_POSTAL_CODE`
* `BROKEN_LOCALITY`
* `BROKEN_REGION`
* `BROKEN_COUNTRY`
* `BROKEN_FULL_ADDRESS`

### Date / time / monetary errors

Use these more carefully. General fields may often be normalized instead of error-coded.

* `BROKEN_ORDER_DATE`
* `BROKEN_ORDER_TIME`
* `BROKEN_CURRENCY`
* `BROKEN_TOTAL`
* `BROKEN_SUBTOTAL`
* `BROKEN_TAX_AMOUNT`

### Product / line-item specificity errors

Use these only when the value is uniquely identifying or would otherwise require guessing.

* `BROKEN_PRODUCT_NAME`
* `BROKEN_SERVICE_NAME`
* `BROKEN_ITEM_CODE`
* `BROKEN_SKU`

### Fallback only when truly necessary

* `BROKEN_UNIQUE_VALUE`

Do not use broad fallback codes when a more precise code is justified.

---

## When to use empty-string unresolved markers

Use an empty string unresolved marker only when:

* the field identity is clear, and
* the value is present-but-unreadable or clearly intended by the document, and
* the value cannot be recovered with sufficient confidence

Examples:

* a clearly labeled phone number line with unreadable digits
* a clearly labeled member ID with OCR corruption
* a clear company-number label with damaged value

Do **not** use empty strings as a lazy substitute for omitted unknown structure.

If both the label and the value are unclear, omit the field entirely.

---

## Omission rule

Prefer omission over fabrication.

Omit the field when:

* the field identity is unclear
* the text might represent several different things
* the OCR fragment does not justify a semantic assignment
* a candidate correction would require guessing
* the document does not actually contain that fact

Do not preserve meaningless garbage merely to fill space.

---

## Evidence-boundary rule

Do not add factual information that is absent from the receipt.

Do not add:

* merchant websites unless printed
* canonical branch names not shown on the receipt
* alternate registration numbers from external knowledge
* inferred city/country/postal details not printed
* corrected product catalog details not supported by the receipt
* external merchant metadata
* geocoding-derived address components
* guessed loyalty program expansions
* guessed cashier identity
* guessed customer identity

External knowledge is not part of the supervision target.

---

## Validation rule

When a value is noisy but plausibly recoverable, you may validate a receipt-supported candidate with web search before committing to the correction.

Web validation is allowed only to confirm a plausible reading already grounded in the receipt.

Web validation must not be used to invent a candidate the receipt never provided.

### Allowed

Receipt says something like `TESC0` and the local evidence strongly suggests `TESCO`.

### Not allowed

Receipt contains an unreadable company blob and the annotator searches for nearby stores to fill it in.

---

## Normalization rules

Normalize when grounded.

### Dates and times

* normalize dates to ISO 8601 where possible
* if both date and time are reliably present, prefer full datetime
* if only date is reliable, use date only
* if the date is clearly present but broken and not recoverable, use an error code rather than guessing

### Currency

* normalize currency to codes such as `EUR`, `USD`, `GBP`, `MYR`
* do not infer a currency merely from merchant identity unless the receipt itself supports it

### Numbers

* normalize machine-readable numbers cleanly
* normalize decimal separators conservatively
* do not coerce broken numeric fragments into a specific value when ambiguous

### Names

* normalize merchant names only when strongly supported
* do not beautify or expand names beyond what the receipt supports

### Addresses

* structure addresses into `PostalAddress` only when supported
* do not split a messy address into precise components unless the structure is actually evidenced

### Product lines

* normalize quantities and prices when possible
* keep product/service names conservative
* do not convert stray OCR fragments into confident product names

### Barcodes and identifiers

* normalize GTIN or barcode values only when the digits are strongly supported
* do not “repair” long identifiers by intuition

---

## Payment and status rules

Use `paymentStatus: "https://schema.org/PaymentComplete"` when the receipt clearly indicates that payment has been completed. `PaymentComplete` is the Schema.org status value for payment that has been received and processed. ([Schema.org][1])

If payment is not clearly complete, do not force `PaymentComplete`.

If order state is clearly represented, use an appropriate order status.

Examples of order-like status situations that may still be represented as `Order`:

* completed / paid
* shipped
* delivered
* pickup available
* pending

Keep status assignment conservative.

---

## Merchant / seller rule

Use the receipt merchant as the seller.

Model the seller as the most appropriate type supported by the receipt:

* `Store` for a clear retail store
* `LocalBusiness` for a specific local merchant
* `Organization` when the receipt only supports organization-level identity

If merchant and branch/store location are both clearly present, the branch/store may be the seller and the parent organization may appear separately if the receipt supports both.

Do not invent a corporate hierarchy.

---

## Line-item modeling rule

Represent each purchased line item as an `Offer` linked to a `Product` or `Service`.

`Order` explicitly allows `acceptedOffer`, and `OrderItem` also exists as a dedicated type for items in an order. Use either pattern or both when they genuinely help express the receipt structure. ([Schema.org][1])

### Minimum acceptable line pattern

* `Offer`
* `price`
* `priceCurrency`
* `itemOffered` as `Product` or `Service` when identifiable

### Better pattern when quantity is available

* `Offer`
* `price`
* `priceCurrency`
* `eligibleQuantity` as `QuantitativeValue`
* `itemOffered`

### Optional richer pattern

You may also use `OrderItem` for a line item node and connect it to the order and/or offered item when that produces cleaner structure.

Do not force `OrderItem` if a simpler `acceptedOffer` structure is enough.

---

## Additional-property rule

Facts that clearly exist on the receipt but have no better natural property should be represented in `additionalProperty` using `PropertyValue`.

Typical cases:

* cashier identifier or label
* terminal number
* lane number
* member number
* store notice code
* payment reference
* ad hoc flags
* rounding line labels
* tax label variants
* loyalty metadata
* receipt footer facts

Use a normalized `name` and a clean `value`.

If the value is broken and unique/sensitive, use an empty `value` with a typed error code.

---

## Document classes that fit as Order

The following input documents are usually in-scope for `Order`:

* paper receipts
* PDF receipts
* order confirmations
* checkout confirmations
* web shop purchase confirmations
* POS purchase records
* pickup-ready notifications
* shipped-order notifications that still describe purchased items
* digital product purchase confirmations
* service purchase confirmations

---

## Document classes that are not primary Order targets

Do not remap these blindly to `Order` unless the text is actually functioning as an order confirmation or purchase record.

### Out of scope or secondary

* pure invoices requesting payment
* bank statements
* standalone shipment tracking notices with no purchase details
* reservations that are fundamentally reservation objects rather than orders
* generic advertisements
* store policies not tied to a transaction
* pure account summaries
* support emails with no transaction confirmation

If a document includes both order data and another concept, the output may contain multiple linked entities as long as `Order` remains the primary transaction representation when appropriate.

---

## Required field behavior

### Include when reliably supported

* `@type: "Order"`
* seller
* order number / receipt number / document number
* order date
* currency
* line items
* totals
* taxes
* discounts
* payment status
* change
* tendered cash
* cashier
* member / loyalty information
* delivery information
* store address
* notices or extra facts via `additionalProperty`

### Do not force when absent or ambiguous

* customer identity
* merchant website
* full postal decomposition
* product brand
* GTIN
* transaction ID
* card details
* country
* branch name
* phone number

---

## Customer rule

If a customer name appears on the receipt but is not explicitly labeled, do not assert it as the customer unless the evidence is strong.

If a person field is clearly present but the value is broken, use an error code instead of guessing.

Do not assume the payer, recipient, buyer, member, and customer are the same person unless the receipt supports that.

---

## Delivery rule

If the document includes delivery or shipping details related to the purchase, model them as order-linked delivery data.

Use:

* `orderDelivery` on the `Order`
* `ParcelDelivery` for shipment or parcel-related details

Possible delivery facts:

* tracking number
* expected arrival
* pickup location
* pickup status
* delivery address

Keep delivery facts tied to actual receipt/order evidence.

---

## Recommended modeling patterns

## Pattern 1: Simple store receipt

Use:

* `Store` or `LocalBusiness` as seller
* `Order`
* `acceptedOffer` array
* `Offer` + `Product`
* totals and currency on the order
* `additionalProperty` for leftovers

## Pattern 2: Rich line-item receipt

Use:

* `Order`
* `acceptedOffer`
* optional `OrderItem`
* `Offer`
* `Product`
* `QuantitativeValue`
* structured totals and tax lines

## Pattern 3: Service receipt

Use:

* `Order`
* `acceptedOffer`
* `Service` as `itemOffered`
* service-specific dates or references in `additionalProperty`

## Pattern 4: Delivery-related order confirmation

Use:

* `Order`
* seller
* `acceptedOffer`
* `orderDelivery`
* `ParcelDelivery`

## Pattern 5: Sparse damaged receipt

Use:

* `Order`
* only the fields truly supported
* unresolved empty values plus error codes for broken specific fields
* omit unsupported details

Sparse output is acceptable. Weak receipts are allowed to produce sparse JSON-LD.

---

## Example: clean minimal receipt

{
"@context": "[https://schema.org](https://schema.org)",
"@graph": [
{
"@id": "#merchant",
"@type": "Store",
"name": "Example Market",
"address": {
"@type": "PostalAddress",
"streetAddress": "123 Example Street",
"addressLocality": "Example City",
"postalCode": "12345",
"addressCountry": "US"
}
},
{
"@id": "#order",
"@type": "Order",
"seller": {
"@id": "#merchant"
},
"orderNumber": "ABC123",
"orderDate": "2024-01-31T14:52:00",
"priceCurrency": "USD",
"paymentStatus": "[https://schema.org/PaymentComplete](https://schema.org/PaymentComplete)",
"acceptedOffer": [
{
"@id": "#offer-1"
}
]
},
{
"@id": "#offer-1",
"@type": "Offer",
"price": "12.99",
"priceCurrency": "USD",
"eligibleQuantity": {
"@type": "QuantitativeValue",
"value": 1,
"unitText": "pc"
},
"itemOffered": {
"@type": "Product",
"name": "Example Product"
}
}
]
}

---

## Example: broken specific value handled correctly

{
"@context": "[https://schema.org](https://schema.org)",
"@graph": [
{
"@id": "#merchant",
"@type": "Store",
"name": "Example Store",
"additionalProperty": [
{
"@type": "PropertyValue",
"name": "merchantPhone",
"value": "",
"description": "ERROR_CODE:BROKEN_PHONE_NUMBER"
},
{
"@type": "PropertyValue",
"name": "businessId",
"value": "",
"description": "ERROR_CODE:BROKEN_COMPANY_NUMBER"
}
]
},
{
"@id": "#order",
"@type": "Order",
"seller": {
"@id": "#merchant"
},
"priceCurrency": "EUR"
}
]
}

---

## Example: broken general value that may be normalized

If OCR says something like:

* `T0TAL`
* `CASHH`
* `EUP`
* `2O24-0I-3I`

you may normalize these only when the receipt evidence strongly supports the intended reading.

Examples of acceptable normalization:

* `T0TAL` -> `Total`
* `CASHH` -> `Cash`
* `EUP` -> `EUR` when the local receipt context supports euro
* `2O24-0I-3I` -> `2024-01-31` when the intended digits are strongly supported

Do not normalize uncertain specific strings using this logic.

---

## Example: omit instead of invent

If the OCR contains a fragment like:

* `Mbr x8?1?`
* `St.... A....`
* `Trx ???`

and the field identity or value is unclear, omit it.

Do not create:

* fake member IDs
* fake street names
* fake transaction IDs
* fake labels just to preserve structure

---

## Review checklist for each annotation

Before finalizing the JSON-LD, verify:

* Is the document fundamentally an order/receipt-like transaction record?
* Is `Order` the primary top-level type?
* Did I use more specific Schema.org types where clearly appropriate?
* Did I avoid adding facts absent from the receipt?
* Did I correct only values grounded in the receipt?
* Did I avoid guessing specific unique/sensitive values?
* Did I use typed error codes for broken specific values?
* Did I omit fields whose identity is unclear?
* Is the JSON valid?
* Is the JSON-LD structurally coherent?
* Is the output sparse when the evidence is sparse?

---

## Absolute prohibitions

Never do any of the following:

* hallucinate a merchant name
* hallucinate a phone number
* hallucinate a person name
* hallucinate a company number
* hallucinate a product name from random OCR fragments
* invent missing IDs
* invent missing addresses
* enrich from the web beyond receipt-supported validation
* output placeholder junk such as `"Unknown item"`
* preserve garbage OCR as if it were reliable structured data
* add explanation text outside the JSON-LD
* convert an invoice into the target just because it looks vaguely transaction-shaped

---

## Final objective

Produce the most semantically correct, receipt-grounded, conservative, machine-usable Schema.org JSON-LD possible.

The model should learn this exact behavior:

* fix grounded general OCR noise
* preserve intact specific values
* never invent broken unique values
* use typed error codes for broken specific fields
* omit unsupported facts
* represent receipt/order-like documents primarily as `Order`

---

## Reference material

* Schema.org `Order` defines an order as a confirmation of a transaction and supports accepted offers, order status, payment status, and delivery linkage. ([Schema.org][1])
* JSON-LD 1.1 is the W3C Recommendation for JSON-based linked data, and supports `@context` and graph-oriented linked structures. ([W3C][2])

This version is already usable as your annotation file. The next refinement would be a stripped training-operator edition with the same rules but less prose and more rigid decision blocks.

[1]: https://schema.org/Order?utm_source=chatgpt.com "Order - Schema.org Type"
[2]: https://www.w3.org/TR/json-ld11/?utm_source=chatgpt.com "JSON-LD 1.1"
