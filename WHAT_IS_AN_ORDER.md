Beyond receipts and order confirmations, anything that is fundamentally a confirmation or stateful record of a transaction can usually be presented as a schema.org/Order, as long as the core thing is “customer accepted offer(s), order exists, here is its status/details.” That is the heart of the type definition. schema.org/Order is defined as “a confirmation of a transaction (a receipt),” and it supports things like acceptedOffer, orderedItem, orderNumber, orderDate, orderStatus, paymentStatus, merchant, seller, and delivery linkage through orderDelivery.

So the common document or item shapes that fit are these:

A purchase receipt fits. That is the most explicit case, because Schema.org basically says so in the definition itself.

An order confirmation fits. This is the classic “thanks, your order was placed” document or email, with line items, price, order number, merchant, and sometimes shipping details. Google’s order markup docs use exactly this framing.

A paid order summary or completed checkout record fits. If the document says the customer bought these products/services and payment is complete or pending, that still maps cleanly to Order, because paymentStatus and orderStatus are first-class properties on the type.

A shipping or fulfillment stage order record can fit, but with a wrinkle. The order itself is still an Order, while the shipping event or package-tracking side is represented through orderDelivery and ParcelDelivery. So a “your order has shipped” document often contains an Order plus linked delivery information, rather than being only a delivery object.

A pickup-ready notification can fit for the same reason. OrderStatus includes OrderPickupAvailable, so a store pickup confirmation is very much in-bounds as an order state update.

A digital goods purchase confirmation fits too. Order is not limited to physical products; its orderedItem can be a Product or a Service, and offers in Schema.org can cover things like streaming, repair services, rentals, or other service-like transactions. So confirmations for software, subscriptions, downloadable content, repairs, installations, and similar purchased services can all fit as Order if they are transaction confirmations.

A return or replacement order record can fit if it is still structured as an order transaction. This is a bit more interpretive, but current industry docs around order schemas explicitly include return-order examples, which is consistent with the idea that an order can represent more than the initial sale, as long as it is still an order-shaped transaction record.

What usually does not belong under Order is just as important. A pure invoice/bill is its own thing: schema.org/Invoice is defined as a statement of money due, and Order merely links to it via partOfInvoice. So a document whose main purpose is “please pay this amount” is not your target beast.

Also, a pure parcel tracking notice is not just an Order by itself; that is more properly a ParcelDelivery, though it may be attached to an order. Same with reservations such as flights, hotels, restaurants, and events: those use the reservation family of types, not Order, unless there is a separate merchandise/service purchase transaction being represented.

A useful rule of thumb is this: if the document could naturally answer “what was bought, by whom, from whom, for how much, and what is the status of that transaction?”, it is probably representable as schema.org/Order. If it mainly answers “what is owed?” it drifts to Invoice. If it mainly answers “where is the package?” it drifts to ParcelDelivery. If it mainly answers “what was booked?” it drifts to Reservation.

For dataset hunting, the practical positive classes are:
receipt PDFs, order confirmation emails/PDFs, paid purchase summaries, shipped-order notices, pickup notices, service purchase confirmations, and digital purchase confirmations. Those are the fattest part of the Order target zone.
